from transformers import AutoProcessor, Wav2Vec2ConformerModel
import torch
from utils.audiodec import AudioDec, assign_model
import  os
import torch
import numpy as np
from tqdm import tqdm
import time
import torchaudio
import shutil
import torch.nn.functional as F
import torch.nn as nn

def tokenize_wav(wav_path,audiodec,device,sample_rate=24000):
    wav, sr = torchaudio.load(wav_path)
    print("origial sampling rate",sr)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)    
    with torch.no_grad():
        wav = wav.unsqueeze(1) #C T-> 1 C T  
        wav = wav.float().to(device)
        z = audiodec.tx_encoder.encode(wav)
        idx = audiodec.tx_encoder.quantize(z)
        
        inc = torch.arange(8)*1024
        idx = idx.cpu() - inc.reshape(-1,1)
        return idx.numpy().T

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

wav_path = 'input.wav'
asr_sample_rate = 16000
wav, sr = torchaudio.load(wav_path) #C T-> 1 C T
print("wav shape",wav.shape)  
if sr != asr_sample_rate:
    wav = torchaudio.functional.resample(wav, sr, asr_sample_rate)
wav = wav.squeeze(0)
inputs = processor(wav, sampling_rate=asr_sample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))

model_name = "libritts_v1"
device = 'cpu' 
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device , rx_device=device )
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

prompt_token = tokenize_wav(wav_path,audiodec,device,sample_rate)
print(prompt_token[:4])
print('audio prompt shape',prompt_token.shape)
print("audio prompt dtype",type(prompt_token))

#align ASR output to shape of prompt audio tokens
last_hidden_states_permuted = last_hidden_states.permute(0, 2, 1)
upsampled_hidden_states_tensor_permuted = F.interpolate(last_hidden_states_permuted, size=len(prompt_token), mode='linear',align_corners=True)
upsampled_hidden_states_tensor_permuted = upsampled_hidden_states_tensor_permuted.permute(0, 2, 1)
print("interpolated last hidden state shape",upsampled_hidden_states_tensor_permuted.shape)

# embed audio tokens 
audiodec_embedding = nn.Embedding(num_embeddings=1024, embedding_dim=128)
prompt_token = torch.from_numpy(prompt_token)
embedded_tokens = audiodec_embedding(prompt_token)  # Shape will be [4, 8, 128]
fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), -1)
print("Fused Embeddings Shape:", fused_embeddings.shape)
audio_projection_layer = nn.Linear(in_features=1024, out_features=1024)
asr_projecttion_layer = nn.Linear(in_features=1024, out_features=1024)

projected_audiodec_embedding = audio_projection_layer(fused_embeddings)
projected_asr_embeddings = asr_projecttion_layer(upsampled_hidden_states_tensor_permuted)


print("projected audio shape",projected_audiodec_embedding.shape)
print("projected asr embedding shape",projected_asr_embeddings.shape)

cross_embedded = torch.empty(2 * len(prompt_token), 1024)

# Assign embeddings to even and odd indices
cross_embedded[0::2, :] = projected_asr_embeddings        # ASR embeddings on even indices
cross_embedded[1::2, :] = projected_audiodec_embedding  # AudioDec embeddings on odd indices

print("cross embedding shape",cross_embedded.shape)

from transformers import LlamaModel, LlamaConfig

configuration = LlamaConfig()

configuration.hidden_size = 1024
configuration.num_hidden_layers = 6
configuration.num_attention_heads = 8
configuration.num_key_value_heads = 8
configuration.intermediate_size= 4096


print("configuration",configuration)
model = LlamaModel(configuration)

print(model)




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
for layer in model.layers:
    layer_outputs = layer(hidden_states, position_ids=position_ids)
    hidden_states = layer_outputs[0]  # Only take the output, ignore attention weights

hidden_states = model.norm(hidden_states)

print("Output shape from transformer layers:", hidden_states.shape)



class AutoregressivePredictor(nn.Module):
    def __init__(self, config):
        super(AutoregressivePredictor, self).__init__()
        self.acoustic_predictor_model = LlamaModel(config)
        self.embedding = nn.Embedding(1024, 256)
        self.output_layer = nn.Linear(config.hidden_size, 1024)

    def forward(self, initial_hidden_state , inference=True):
        current_input = initial_hidden_state
        predicted_tokens = []
        
        if inference:
            for position in range(8):  # Generate 8 tokens autoregressively
                # Forward pass through the model
                outputs = self.acoustic_predictor_model.layers[0](current_input,position_ids = torch.arange(0, current_input.shape[1]).unsqueeze(0))
                print("outputs shape",outputs[0].shape)
                #logits = outputs[0].last_hidden_state[:, -1, :]
                transformed_logits = self.output_layer(outputs[0][:, -1, :])
                print("transformed logit",transformed_logits.shape)
                probabilities = F.softmax(transformed_logits, dim=-1)
                predicted_token = torch.argmax(probabilities, dim=-1)
                predicted_tokens.append(predicted_token)
                #logits = outputs[0].logits  # Shape: [batch_size, seq_length, vocab_size]
                # Prepare the input for the next step
                token_embedding = self.embedding(predicted_token)
                print("current input shape",current_input.shape,"token embedding shape",token_embedding.shape)
                token_embedding = token_embedding.unsqueeze(0)
                current_input = torch.cat([current_input, token_embedding], dim=1)
        #print("predicted tokens",predicted_tokens)
        return predicted_tokens  # Shape: [batch_size, 8]

# Configuration for the acoustic predictor
acoustic_configuration = LlamaConfig()

acoustic_configuration.hidden_size = 256
acoustic_configuration.num_hidden_layers = 1
acoustic_configuration.num_attention_heads = 4
acoustic_configuration.num_key_value_heads = 4
acoustic_configuration.intermediate_size= 1024

# Initialize the acoustic predictor module
ar_predictor = AutoregressivePredictor(acoustic_configuration)

# Example initial hidden state
initial_hidden_state = torch.randn(1, 1, 256)  # [batch_size, 1 (initial sequence length), hidden_size]

# Perform autoregressive prediction
predicted_sequence = ar_predictor(initial_hidden_state)
print("Predicted sequence of tokens:", predicted_sequence)

















