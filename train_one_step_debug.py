import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ConformerModel, LlamaModel, LlamaConfig
from utils.audiodec import AudioDec, assign_model
import torch.optim as optim
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

'''layer Transformer decoder
with a hidden size 256, feed-forward hidden size
1
https://github.com/facebookresearch/AudioDec
2
https://github.com/b04901014/MQTTS
1024, and 4 heads'''

class AutoregressivePredictor(nn.Module):
    def __init__(self):
        super(AutoregressivePredictor, self).__init__()
        #self.acoustic_predictor_model = LlamaModel(config)
        llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B") 
        self.acoustic_predictor_model = llama_model.model.layers[0]
        self.norm =  llama_model.model.norm
        self.embedding = nn.Embedding(1024, 4096)
        self.output_layer = nn.Linear(4096, 1024)
        #self.arp_projection = nn.Linear(1024, 256)

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
            hidden_states = chunk_hidden_states
            #for layer in self.acoustic_predictor_model.model.layers[-1]:
            layer_outputs = self.acoustic_predictor_model(hidden_states, position_ids=torch.arange(0, chunk_hidden_states.shape[1]).unsqueeze(0).to('cuda'))
            hidden_states = layer_outputs[
                    0
                ]  # Only take the output, ignore attention weights
            hidden_states = self.norm(hidden_states)    
            transformed_logits = self.output_layer(hidden_states)
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
            token_embedding = self.embedding(predicted_token).unsqueeze(0)
            current_input = torch.cat([current_input, token_embedding], dim=1)

        return predicted_tokens


class CyborgEncoder(nn.Module):
    def __init__(self, audiodec):
        super(CyborgEncoder, self).__init__()
        self.audiodec = audiodec
        self.audiodec_embedding = nn.Embedding(num_embeddings=1024, embedding_dim=1024)
        self.audio_projection_layer = nn.Linear(in_features=8192, out_features=4096)
        # bottleneck regulator
        # Configure the Llama model for ASR embeddings
        '''llama_config = LlamaConfig(
            hidden_size=1024,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=4096,
        )'''
        

        #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
        print("llama model",self.llama_model)
        
        
        #self.llama_model = LlamaModel(llama_config)
        '''self.ar_predictor_config = LlamaConfig(
            hidden_size=4096,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=4096,
        )'''
        #self.ar_predictor = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

        self.ar_predictor = AutoregressivePredictor()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.k = 4

    def forward(
        self,
        wav_path,
        output_wav_path=None,
        device="cuda",
        asr_sample_rate=16000,
        sample_rate=24000,
        inference=False,
    ):
        prompt_token,wav_input = self.tokenize_wav(wav_path, self.audiodec, device, sample_rate)
        prompt_token = prompt_token.to('cuda')
        #print("Input prompt shape",prompt_token.shape)
        if not inference:
            output_prompt_token,wav_output = self.tokenize_wav(
                output_wav_path, self.audiodec, device, sample_rate
            )
            output_prompt_token = output_prompt_token.to('cuda')
            min_length = min(len(prompt_token), len(output_prompt_token))
            prompt_token, output_prompt_token = (
                prompt_token[:min_length, :],
                output_prompt_token[:min_length, :],
            )
        #self.audiodec_embedding.to('cuda') 
        embedded_tokens = self.audiodec_embedding(prompt_token)
        fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), -1)

        projected_audiodec_embedding = self.audio_projection_layer(fused_embeddings)
        # bottle neck regulator
        # AudioDec embeddings on odd indices
        cross_embedded = projected_audiodec_embedding
        if cross_embedded.dim() == 2:
            cross_embedded = cross_embedded.unsqueeze(
                0
            )  # Add batch dimension if needed
        sequence_length = cross_embedded.size(1)
        position_ids = (
            torch.arange(sequence_length).unsqueeze(0).repeat(cross_embedded.size(0), 1)
        ).to('cuda')  # repeat for batch size
        # Forward pass through the model
        # Initialize hidden states variable, this will be updated with each layer's output
        hidden_states = cross_embedded
        #hidden_states = self.llama_model.model.layers(hidden_states, position_ids=position_ids)[0]
        for layer in self.llama_model.model.layers:
            layer_outputs = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[
                0
            ]  # Only take the output, ignore attention weights

        hidden_states = self.llama_model.model.norm(hidden_states)
        if inference:
            predicted_sequence = self.ar_predictor(hidden_states, inference=True)
            return predicted_sequence,wav_input
        else:
            hidden_states = hidden_states.permute(1, 0, 2)
            target_output_token_tensor = output_prompt_token
            output_prompt_token_tensor = output_prompt_token[:, :7]
            
            output_prompt_token_embeddings = self.ar_predictor.embedding(
                output_prompt_token_tensor
            )  # [sequence_length, 7, embedding_size]
            #project hidden states 
            #hidden_states = self.ar_predictor.arp_projection(hidden_states)
            #print("projectes hidden states shape",hidden_states.shape)
            chunk_tensor = torch.cat(
                (hidden_states, output_prompt_token_embeddings), dim=1
            )
            predicted_logits = self.ar_predictor(chunk_tensor, inference=False)
            predicted_tokens = torch.argmax(predicted_logits, dim=2)
            print("predicted tokens",predicted_tokens)
            print("target prompt tokens",output_prompt_token)
            with torch.no_grad():
                #create the audio file
                predicted_tokens = predicted_tokens.T
                inc = torch.arange(8) * 1024
                inc = inc.to('cuda')
                predicted_idx = (predicted_tokens + inc.reshape(-1, 1))
                predicted_zq = audiodec.rx_encoder.lookup(predicted_idx)

                y = audiodec.decoder.decode(predicted_zq)[:, :, :wav_input.size(-1)]
                y = y.squeeze(1).transpose(1, 0).cpu().numpy() # T x C
                sf.write(
                    'predicted_train_audio.wav',
                    y,
                    24000,
                    "PCM_16",
                )

            one_hot_encoded = F.one_hot(
                target_output_token_tensor, num_classes=1024
            ).float()
            #print("predicted logits shape",predicted_logits.shape)
            #print("one hot encoded shape",one_hot_encoded.shape)
            cross_entropy_loss = self.cross_entropy_loss_fn(
                predicted_logits, one_hot_encoded
            )
            '''teacher_force_loss = self.calculate_tf_loss(
                context_hidden_state, upsampled_hidden_states_tensor_permuted
            )'''
            total_loss = cross_entropy_loss #+ 0.1*teacher_force_loss
            return predicted_logits,total_loss,wav_input

    def tokenize_wav(self, wav_path, audiodec, device, sample_rate):
        wav, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.unsqueeze(1).float().to(device)
        with torch.no_grad():
            z = audiodec.tx_encoder.encode(wav)
            idx = audiodec.tx_encoder.quantize(z)
        inc = torch.arange(8) * 1024
        inc = inc.to('cuda')
        idx = idx.to('cuda')
        idx = (idx - inc.reshape(-1, 1)).T
        return idx,wav

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



model_name = "libritts_v1"
device = "cuda"
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

cyborg_encoder = CyborgEncoder(audiodec)
cyborg_encoder.to('cuda')
# Setup the AdamW optimizer
optimizer = optim.AdamW(cyborg_encoder.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.916)  # Exponential decay factor

def train_model(model, wav_path, output_wav_path, device, optimizer,step, accumulation_steps=1):
    model.train()  # Set model to training mode
    total_loss = 0

    # Forward pass: Compute predicted loss by passing inputs to the model
    predicted_sequences, loss, wav_input = model(wav_path, output_wav_path, device=device, inference=False)
    loss = loss / accumulation_steps  # Normalize loss to account for the accumulation of gradients

    # Backward pass: Perform backpropagation to calculate gradients
    loss.backward()
    total_loss += loss.item()

    # Only step the optimizer and zero gradients every 'accumulation_steps'
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return total_loss


def save_model_only(model, path="cyborg_encoder_model.pth"):
    """Saves only the model's state dictionary."""
    torch.save(model.state_dict(), path)
    print(f"Model saved at '{path}'")


wav_path = "vcc_input.wav"
output_wav_path = "vcc_output.wav"



# Train the model for 1000 steps
import os
file_list = os.listdir("/workspace/VCC2020-database/target_task1/TEF1")
file_list = file_list[:1]
num_steps = 50000
epochs = 10
step = 0
while step<num_steps:
    for file_name in file_list:
        input_wav_path = os.path.join("/workspace/VCC2020-database/target_task1/TEF2",file_name)
        output_wav_path = os.path.join("/workspace/VCC2020-database/target_task1/TEF1",file_name)

        loss = train_model(cyborg_encoder, input_wav_path, output_wav_path, device, optimizer,step)
        print("file name",file_name)
        if step % 1 == 0:  # Print the loss every 10 steps
            print(f"Step {step}/{num_steps}, Loss: {loss}")
            save_model_only(cyborg_encoder)
        step = step + 1 #update number of steps    
    #  # Update the learning rate after each epoch
    print(f"Learning rate updated to {scheduler.get_last_lr()}")

print("Training complete.")
print(file_list)
#test inference 
saved_path="cyborg_encoder_model.pth"
cyborg_encoder.load_state_dict(torch.load(saved_path))
cyborg_encoder.eval()  # Set the model to evaluation mode
predicted_sequences,wav_input = cyborg_encoder(wav_path, inference=True)
with torch.no_grad():
    #create the audio file
    predicted_sequences = predicted_sequences.T
    inc = torch.arange(8) * 1024
    predicted_idx = (predicted_sequences + inc.reshape(-1, 1))
    predicted_zq = audiodec.rx_encoder.lookup(predicted_idx)

    y = audiodec.decoder.decode(predicted_zq)[:, :, :wav_input.size(-1)]
    y = y.squeeze(1).transpose(1, 0).cpu().numpy() # T x C
    sf.write(
        'predicted_audio.wav',
        y,
        24000,
        "PCM_16",
    )


#print(f"Model loaded from '{path}' and set to evaluation mode")




