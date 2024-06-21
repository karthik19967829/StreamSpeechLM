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

    def forward(self, chunk_hidden_states , inference=True):
        chunk_predicted_tokens = []
        if inference:
            print("total time steps",chunk_hidden_states.size(1))
            for time_index in range(chunk_hidden_states.size(1)):
                current_input = chunk_hidden_states[:,time_index,:].unsqueeze(0)
                timestamp_predicted_tokens = []
        
                for position in range(8):  # Generate 8 tokens autoregressively
                    # Forward pass through the model
                    outputs = self.acoustic_predictor_model.layers[0](current_input,position_ids = torch.arange(0, current_input.shape[1]).unsqueeze(0))
                    print("outputs shape",outputs[0].shape)
                    #logits = outputs[0].last_hidden_state[:, -1, :]
                    transformed_logits = self.output_layer(outputs[0][:, -1, :])
                    print("transformed logit",transformed_logits.shape)
                    probabilities = F.softmax(transformed_logits, dim=-1)
                    predicted_token = torch.argmax(probabilities, dim=-1)
                    timestamp_predicted_tokens.append(predicted_token.item())
                    #logits = outputs[0].logits  # Shape: [batch_size, seq_length, vocab_size]
                    # Prepare the input for the next step
                    token_embedding = self.embedding(predicted_token)
                    print("current input shape",current_input.shape,"token embedding shape",token_embedding.shape)
                    token_embedding = token_embedding.unsqueeze(0)
                    current_input = torch.cat([current_input, token_embedding], dim=1)
                # append this time stamps to prediction to chunk level 
                chunk_predicted_tokens.append(timestamp_predicted_tokens)     
        #print("predicted tokens",predicted_tokens)
        return chunk_predicted_tokens  # Shape: [batch_size, 8]

class CyborgEncoder(nn.Module):
    def __init__(self, asr_processor, asr_model, audiodec):
        super(CyborgEncoder, self).__init__()
        self.asr_processor = asr_processor
        self.asr_model = asr_model
        self.audiodec = audiodec
        self.audiodec_embedding = nn.Embedding(num_embeddings=1024, embedding_dim=128)
        self.audio_projection_layer = nn.Linear(in_features=1024, out_features=1024)
        self.asr_projection_layer = nn.Linear(in_features=1024, out_features=1024)

        
        # Configure the Llama model for ASR embeddings
        llama_config = LlamaConfig(
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=4096
        )
        self.llama_model = LlamaModel(llama_config)
        self.ar_predictor_config = LlamaConfig(
            hidden_size=1024,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=4096
        )
        self.ar_predictor = AutoregressivePredictor(self.ar_predictor_config)
        
    def forward(self, wav_path, device='cpu', asr_sample_rate=16000, sample_rate=24000):
        wav, sr = torchaudio.load(wav_path)
        if sr != asr_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, asr_sample_rate)
        wav = wav.squeeze(0)
        inputs = self.asr_processor(wav, sampling_rate=asr_sample_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.asr_model(**inputs)
        prompt_token = self.tokenize_wav(wav_path, self.audiodec, device, sample_rate)    
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states_permuted = last_hidden_states.permute(0, 2, 1)
        upsampled_hidden_states_tensor_permuted = F.interpolate(last_hidden_states_permuted, size=len(prompt_token), mode='linear',align_corners=True)
        upsampled_hidden_states_tensor_permuted = upsampled_hidden_states_tensor_permuted.permute(0, 2, 1)
        
        embedded_tokens = self.audiodec_embedding(torch.from_numpy(prompt_token))
        fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), -1)
        
        projected_audiodec_embedding = self.audio_projection_layer(fused_embeddings)
        projected_asr_embeddings = self.asr_projection_layer(upsampled_hidden_states_tensor_permuted)
        
        cross_embedded = torch.empty(2 * len(prompt_token), 1024)

        # Assign embeddings to even and odd indices
        cross_embedded[0::2, :] = projected_asr_embeddings        # ASR embeddings on even indices
        cross_embedded[1::2, :] = projected_audiodec_embedding  # AudioDec embeddings on odd indices
        if cross_embedded.dim() == 2:
            cross_embedded = cross_embedded.unsqueeze(0)  # Add batch dimension if needed
        sequence_length = cross_embedded.size(1)
        print("sequence lenght",sequence_length)
        position_ids = torch.arange(sequence_length).unsqueeze(0).repeat(cross_embedded.size(0), 1)  # repeat for batch size
        # Forward pass through the model
        # Initialize hidden states variable, this will be updated with each layer's output
        hidden_states = cross_embedded
        print("position ids",position_ids.shape)
        print("hidden states",hidden_states.shape)
        for layer in self.llama_model.layers:
            layer_outputs = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[0]  # Only take the output, ignore attention weights

        hidden_states = self.llama_model.norm(hidden_states) 
        print("hidden states 0 shape",hidden_states.shape)
        predicted_sequence = self.ar_predictor(hidden_states)
        return predicted_sequence
    
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

# Usage
asr_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
asr_model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model_name = "libritts_v1"
device = 'cpu'
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

cyborg_encoder = CyborgEncoder(asr_processor, asr_model, audiodec)
wav_path = 'input.wav'
output = cyborg_encoder(wav_path)
print(output)
