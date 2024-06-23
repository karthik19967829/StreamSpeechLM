import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ConformerModel, LlamaModel, LlamaConfig
from utils.audiodec import AudioDec, assign_model


class AutoregressivePredictor(nn.Module):
    def __init__(self, config):
        super(AutoregressivePredictor, self).__init__()
        self.acoustic_predictor_model = LlamaModel(config)
        self.embedding = nn.Embedding(1024, 1024)
        self.output_layer = nn.Linear(config.hidden_size, 1024)

    def forward(self, chunk_hidden_states, inference=True):
        if inference:
            num_time_steps = chunk_hidden_states.size(1)
            # Initialize tensor to hold all predicted tokens
            chunk_predicted_tokens = torch.empty((num_time_steps, 8), dtype=torch.long)
            # during final inference optimization , need to run this across multiple process cores
            for time_index in range(num_time_steps):
                current_input = chunk_hidden_states[:, time_index, :].unsqueeze(0)
                timestamp_predicted_tokens = self.autoregressively_generate_tokens(
                    current_input
                )
                chunk_predicted_tokens[time_index, :] = timestamp_predicted_tokens

            return chunk_predicted_tokens

        else:
            outputs = self.acoustic_predictor_model.layers[0](
                chunk_hidden_states,
                position_ids=torch.arange(0, chunk_hidden_states.shape[1]).unsqueeze(0),
            )
            transformed_logits = self.output_layer(outputs[0])
            transformed_logits = transformed_logits.view(
                -1, 8, 1024
            )  # Ensure logits are shaped as (num_timesteps, 8, 1024)
            return transformed_logits

    def autoregressively_generate_tokens(self, current_input):
        predicted_tokens = torch.empty(
            8, dtype=torch.long
        )  # Tensor for the predicted tokens

        for position in range(8):  # Autoregressive generation of 8 tokens
            outputs = self.acoustic_predictor_model.layers[0](
                current_input,
                position_ids=torch.arange(0, current_input.shape[1]).unsqueeze(0),
            )
            transformed_logits = self.output_layer(outputs[0][:, -1, :])
            probabilities = F.softmax(transformed_logits, dim=-1)
            predicted_token = torch.argmax(probabilities, dim=-1)
            predicted_tokens[position] = (
                predicted_token  # Store directly into the tensor
            )

            # Get token embedding and prepare for the next step
            token_embedding = self.embedding(predicted_token).unsqueeze(0)
            current_input = torch.cat([current_input, token_embedding], dim=1)

        return predicted_tokens


class CyborgEncoder(nn.Module):
    def __init__(self, asr_processor, asr_model, audiodec):
        super(CyborgEncoder, self).__init__()
        self.asr_processor = asr_processor
        self.asr_model = asr_model
        self.audiodec = audiodec
        self.audiodec_embedding = nn.Embedding(num_embeddings=1024, embedding_dim=128)
        self.audio_projection_layer = nn.Linear(in_features=1024, out_features=1024)
        # bottleneck regulator
        self.asr_down_projection_layer = nn.Linear(in_features=1024, out_features=512)
        self.asr_upward_projection_layer = nn.Linear(in_features=512, out_features=1024)
        self.context_linear_prediction = nn.Linear(in_features=1024, out_features=1024)

        # Configure the Llama model for ASR embeddings
        llama_config = LlamaConfig(
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=4096,
        )
        self.llama_model = LlamaModel(llama_config)
        self.ar_predictor_config = LlamaConfig(
            hidden_size=1024,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=4096,
        )
        self.ar_predictor = AutoregressivePredictor(self.ar_predictor_config)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.k = 4

    def forward(
        self,
        wav_path,
        output_wav_path,
        device="cpu",
        asr_sample_rate=16000,
        sample_rate=24000,
        inference=False,
    ):
        wav, sr = torchaudio.load(wav_path)
        if sr != asr_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, asr_sample_rate)
        wav = wav.squeeze(0)
        inputs = self.asr_processor(
            wav, sampling_rate=asr_sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.asr_model(**inputs)
        prompt_token = self.tokenize_wav(wav_path, self.audiodec, device, sample_rate)
        if not inference:
            output_prompt_token = self.tokenize_wav(
                output_wav_path, self.audiodec, device, sample_rate
            )
            min_length = min(len(prompt_token), len(output_prompt_token))
            prompt_token, output_prompt_token = (
                prompt_token[:min_length, :],
                output_prompt_token[:min_length, :],
            )
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states_permuted = last_hidden_states.permute(0, 2, 1)
        upsampled_hidden_states_tensor_permuted = F.interpolate(
            last_hidden_states_permuted,
            size=len(prompt_token),
            mode="linear",
            align_corners=True,
        )
        upsampled_hidden_states_tensor_permuted = (
            upsampled_hidden_states_tensor_permuted.permute(0, 2, 1)
        )

        embedded_tokens = self.audiodec_embedding(torch.from_numpy(prompt_token))
        fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), -1)

        projected_audiodec_embedding = self.audio_projection_layer(fused_embeddings)
        # bottle neck regulator
        down_projected_context_hidden_state = self.asr_down_projection_layer(
            upsampled_hidden_states_tensor_permuted
        )
        projected_asr_embeddings = self.asr_upward_projection_layer(
            down_projected_context_hidden_state
        )
        cross_embedded = torch.empty(2 * len(prompt_token), 1024)

        # Assign embeddings to even and odd indices
        cross_embedded[0::2, :] = (
            projected_asr_embeddings  # ASR embeddings on even indices
        )
        cross_embedded[1::2, :] = (
            projected_audiodec_embedding  # AudioDec embeddings on odd indices
        )
        if cross_embedded.dim() == 2:
            cross_embedded = cross_embedded.unsqueeze(
                0
            )  # Add batch dimension if needed
        sequence_length = cross_embedded.size(1)
        position_ids = (
            torch.arange(sequence_length).unsqueeze(0).repeat(cross_embedded.size(0), 1)
        )  # repeat for batch size
        # Forward pass through the model
        # Initialize hidden states variable, this will be updated with each layer's output
        hidden_states = cross_embedded
        for layer in self.llama_model.layers:
            layer_outputs = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[
                0
            ]  # Only take the output, ignore attention weights

        hidden_states = self.llama_model.norm(hidden_states)
        # get the semantic hidden states
        hidden_states = hidden_states[:, 0::2, :]
        # semantic contenxtual the hidden states
        context_hidden_state = self.context_linear_prediction(hidden_states)
        # bottle neck regulator
        down_projected_context_hidden_state = self.asr_down_projection_layer(
            context_hidden_state
        )
        upward_projected_context_hidden_state = self.asr_upward_projection_layer(
            down_projected_context_hidden_state
        )
        # add this to hidden states
        hidden_states = hidden_states + upward_projected_context_hidden_state
        if inference:
            predicted_sequence = self.ar_predictor(hidden_states, inference=True)
            return predicted_sequence
        else:
            hidden_states = hidden_states.permute(1, 0, 2)
            target_output_token_tensor = torch.from_numpy(output_prompt_token)
            output_prompt_token_tensor = torch.from_numpy(
                output_prompt_token[:, :7]
            ).long()
            output_prompt_token_embeddings = self.ar_predictor.embedding(
                output_prompt_token_tensor
            )  # [sequence_length, 7, embedding_size]
            chunk_tensor = torch.cat(
                (hidden_states, output_prompt_token_embeddings), dim=1
            )
            predicted_logits = self.ar_predictor(chunk_tensor, inference=False)
            one_hot_encoded = F.one_hot(
                target_output_token_tensor, num_classes=1024
            ).float()
            cross_entropy_loss = self.cross_entropy_loss_fn(
                predicted_logits, one_hot_encoded
            )
            teacher_force_loss = self.calculate_tf_loss(
                context_hidden_state, upsampled_hidden_states_tensor_permuted
            )
            total_loss = cross_entropy_loss + teacher_force_loss
            return total_loss

    def tokenize_wav(self, wav_path, audiodec, device, sample_rate):
        wav, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.unsqueeze(1).float().to(device)
        with torch.no_grad():
            z = audiodec.tx_encoder.encode(wav)
            idx = audiodec.tx_encoder.quantize(z)
        inc = torch.arange(8) * 1024
        idx = (idx.cpu() - inc.reshape(-1, 1)).numpy().T
        return idx

    def calculate_tf_loss(self, context_vectors, semantic_features):
        batch_size, time_steps, _ = semantic_features.shape
        losses = []
        # Loop over time to calculate loss for each timestep considering future steps up to t+k
        for t in range(time_steps - self.k):
            # Ground truth context is concatenation of semantic features from t to t+k
            ground_truth = semantic_features[:, t : t + self.k].reshape(
                batch_size, -1
            )  # Reshape to [batch_size, (k+1)*feature_dim]
            prediction = (
                context_vectors[:, t]
                .unsqueeze(1)
                .repeat(1, self.k, 1)
                .reshape(batch_size, -1)
            )

            # Calculate MSE loss
            mse_loss = F.mse_loss(prediction, ground_truth)
            losses.append(mse_loss)

        # Average loss over all considered timesteps
        tf_total_loss = torch.mean(torch.stack(losses))
        return tf_total_loss


# Usage
asr_processor = AutoProcessor.from_pretrained(
    "facebook/wav2vec2-conformer-rope-large-960h-ft"
)
asr_model = Wav2Vec2ConformerModel.from_pretrained(
    "facebook/wav2vec2-conformer-rope-large-960h-ft"
)
model_name = "libritts_v1"
device = "cpu"
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

cyborg_encoder = CyborgEncoder(asr_processor, asr_model, audiodec)
wav_path = "input.wav"
loss = cyborg_encoder(wav_path, wav_path)
print("total loss", loss)
