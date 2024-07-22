import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ConformerModel, LlamaModel, LlamaConfig
from utils.audiodec import AudioDec, assign_model
import torch.optim as optim
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

class AutoregressivePredictor(nn.Module):
    def __init__(self):
        super(AutoregressivePredictor, self).__init__()
        llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
        self.acoustic_predictor_model = llama_model.model.layers[0]
        self.norm = llama_model.model.norm
        self.embedding = nn.Embedding(1024, 4096)
        self.output_layer = nn.Linear(4096, 1024, dtype=torch.bfloat16)  # Ensure output layer uses bfloat16

    def forward(self, chunk_hidden_states, inference=True):
        chunk_hidden_states = chunk_hidden_states.to(torch.bfloat16)
        if inference:
            num_time_steps = chunk_hidden_states.size(1)
            batch_size = chunk_hidden_states.size(0)
            chunk_predicted_tokens = torch.empty((batch_size, num_time_steps, 8), dtype=torch.long)
            for time_index in range(num_time_steps):
                current_input = chunk_hidden_states[:, time_index, :].unsqueeze(1)
                timestamp_predicted_tokens = self.autoregressively_generate_tokens(current_input)
                chunk_predicted_tokens[:, time_index, :] = timestamp_predicted_tokens
            return chunk_predicted_tokens
        else:
            hidden_states = chunk_hidden_states
            batch_size = hidden_states.size(0)
            position_ids = torch.arange(0, hidden_states.shape[1], dtype=torch.bfloat16).unsqueeze(0).repeat(batch_size, 1).to('cuda')
            layer_outputs = self.acoustic_predictor_model(hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states).to(torch.bfloat16)  # Ensure norm output is bfloat16
            transformed_logits = self.output_layer(hidden_states)  # Ensure hidden states are bfloat16
            transformed_logits = transformed_logits.view(batch_size, -1, 8, 1024)
            return transformed_logits

    def autoregressively_generate_tokens(self, current_input):
        current_input = current_input.to(torch.bfloat16)
        batch_size = current_input.size(0)
        predicted_tokens = torch.empty(batch_size, 8, dtype=torch.long)
        for position in range(8):
            position_ids = torch.arange(0, current_input.shape[1], dtype=torch.bfloat16).unsqueeze(0).repeat(batch_size, 1).to('cuda')
            outputs = self.acoustic_predictor_model(current_input, position_ids=position_ids)
            transformed_logits = self.output_layer(outputs[0][:, -1, :])
            probabilities = F.softmax(transformed_logits, dim=-1)
            predicted_token = torch.argmax(probabilities, dim=-1)
            predicted_tokens[:, position] = predicted_token
            token_embedding = self.embedding(predicted_token).unsqueeze(1).to(current_input.dtype)
            current_input = torch.cat([current_input, token_embedding], dim=1)
        return predicted_tokens

class CyborgEncoder(nn.Module):
    def __init__(self, audiodec):
        super(CyborgEncoder, self).__init__()
        self.audiodec = audiodec
        self.audiodec_embedding = nn.Embedding(num_embeddings=1024, embedding_dim=1024)
        self.audio_projection_layer = nn.Linear(in_features=8192, out_features=4096, dtype=torch.bfloat16)
        self.llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16)
        self.ar_predictor = AutoregressivePredictor()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.k = 4

    def forward(self, wav_paths, output_wav_paths=None, device="cuda", asr_sample_rate=16000, sample_rate=24000, inference=False):
        prompt_tokens, wav_inputs = self.tokenize_wav(wav_paths, self.audiodec, device, sample_rate)
        prompt_tokens = prompt_tokens.to(device, dtype=torch.long)  # Ensure indices are of type Long
        if not inference:
            output_prompt_tokens, wav_outputs = self.tokenize_wav(output_wav_paths, self.audiodec, device, sample_rate)
            output_prompt_tokens = output_prompt_tokens.to(device, dtype=torch.long)  # Ensure indices are of type Long
            min_length = min(prompt_tokens.size(1), output_prompt_tokens.size(1))
            prompt_tokens, output_prompt_tokens = prompt_tokens[:, :min_length], output_prompt_tokens[:, :min_length]
        else:
            min_length = prompt_tokens.size(1)
        embedded_tokens = self.audiodec_embedding(prompt_tokens).to(torch.bfloat16)
        fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), embedded_tokens.size(1), -1)
        projected_audiodec_embedding = self.audio_projection_layer(fused_embeddings)
        '''print("emebdded tokens shape",embedded_tokens.shape)
        fused_embeddings = embedded_tokens.view(embedded_tokens.size(0), -1)
        #fused_embeddings = embedded_tokens[:, :min_length, :]
        projected_audiodec_embedding = self.audio_projection_layer(fused_embeddings)'''
        cross_embedded = projected_audiodec_embedding
        sequence_length = cross_embedded.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.bfloat16).unsqueeze(0).repeat(cross_embedded.size(0), 1).to('cuda')
        hidden_states = cross_embedded
        for layer in self.llama_model.model.layers:
            hidden_states = hidden_states.to(torch.bfloat16)  # Ensure hidden_states are in bfloat16
            print("hidden states shape line 87",hidden_states.shape)
            layer_outputs = layer(hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[0]
        hidden_states = self.llama_model.model.norm(hidden_states).to(torch.bfloat16)  # Ensure norm output is bfloat16
        if inference:
            predicted_sequence = self.ar_predictor(hidden_states, inference=True)
            return predicted_sequence[:, :min_length], wav_inputs
        else:
            print("hidden states shape",hidden_states.shape)
            hidden_states = hidden_states.permute(1, 0, 2)
            target_output_token_tensor = output_prompt_tokens
            output_prompt_token_tensor = output_prompt_tokens[:, :7]
            output_prompt_token_embeddings = self.ar_predictor.embedding(output_prompt_token_tensor)
            print("output prompt token embeddings 104",output_prompt_token_embeddings.shape)
            print("hidden states 105",hidden_states.shape)
            chunk_tensor = torch.cat((hidden_states, output_prompt_token_embeddings), dim=1)
            predicted_logits = self.ar_predictor(chunk_tensor, inference=False)
            predicted_tokens = torch.argmax(predicted_logits, dim=2)
            with torch.no_grad():
                predicted_tokens = predicted_tokens.permute(1, 0)
                inc = torch.arange(8) * 1024
                inc = inc.to(device)
                predicted_idx = (predicted_tokens + inc.reshape(-1, 1))
                predicted_zq = self.audiodec.rx_encoder.lookup(predicted_idx)
                y = self.audiodec.decoder.decode(predicted_zq)[:, :, :wav_inputs.size(-1)]
                y = y.squeeze(1).transpose(1, 0).cpu().numpy()
                sf.write('predicted_train_audio.wav', y, 24000, "PCM_16")
            one_hot_encoded = F.one_hot(target_output_token_tensor, num_classes=1024).float().to(device, dtype=torch.bfloat16)
            cross_entropy_loss = self.cross_entropy_loss_fn(predicted_logits[:, :min_length], one_hot_encoded[:, :min_length])
            total_loss = cross_entropy_loss
            return predicted_logits[:, :min_length], total_loss, wav_inputs

    def tokenize_wav(self, wav_paths, audiodec, device, sample_rate):
        batch_tokenized = []
        batch_wav = []
        for wav_path in wav_paths:
            wav, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            wav = wav.unsqueeze(1).float().to(device)
            with torch.no_grad():
                z = audiodec.tx_encoder.encode(wav)
                idx = audiodec.tx_encoder.quantize(z)
            inc = torch.arange(8) * 1024
            inc = inc.to(device)
            idx = idx.to(device, dtype=torch.long)  # Ensure indices are of type Long
            idx = (idx - inc.reshape(-1, 1)).T
            batch_tokenized.append(idx)
            batch_wav.append(wav)
        min_length = min(map(len, batch_tokenized))
        batch_tokenized = torch.stack([t[:min_length] for t in batch_tokenized])
        batch_wav = torch.stack([w[:, :min_length] for w in batch_wav])
        return batch_tokenized, batch_wav

model_name = "libritts_v1"
device = "cuda"
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

cyborg_encoder = CyborgEncoder(audiodec)
cyborg_encoder.to(device)

optimizer = optim.AdamW(cyborg_encoder.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.916)

def train_model(model, wav_paths, output_wav_paths, device, optimizer, step, accumulation_steps=32):
    model.train()
    total_loss = 0
    predicted_sequences, loss, wav_input = model(wav_paths, output_wav_paths, device=device, inference=False)
    loss = loss / accumulation_steps
    loss.backward()
    total_loss += loss.item()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    return total_loss

def save_model_only(model, path="cyborg_encoder_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at '{path}'")

wav_paths = ["vcc_input1.wav", "vcc_input2.wav", "vcc_input3.wav", "vcc_input4.wav", "vcc_input5.wav", "vcc_input6.wav", "vcc_input7.wav"]
output_wav_paths = ["vcc_output1.wav", "vcc_output2.wav", "vcc_output3.wav", "vcc_output4.wav", "vcc_output5.wav", "vcc_output6.wav", "vcc_output7.wav"]

import os
file_list = os.listdir("/workspace/VCC2020-database/target_task1/TEF1")
file_list = file_list[:7]
num_steps = 50000
epochs = 10
step = 0
while step < num_steps:
    for file_name in file_list:
        input_wav_paths = [os.path.join("/workspace/VCC2020-database/target_task1/TEF2", file_name)] * 7
        output_wav_paths = [os.path.join("/workspace/VCC2020-database/target_task1/TEF1", file_name)] * 7
        loss = train_model(cyborg_encoder, input_wav_paths, output_wav_paths, device, optimizer, step)
        if step % 1 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss}")
            save_model_only(cyborg_encoder)
        step += 1
    print(f"Learning rate updated to {scheduler.get_last_lr()}")

print("Training complete.")

saved_path = "cyborg_encoder_model.pth"
cyborg_encoder.load_state_dict(torch.load(saved_path))
cyborg_encoder.eval()
predicted_sequences, wav_input = cyborg_encoder(wav_paths, inference=True)
with torch.no_grad():
    predicted_sequences = predicted_sequences.permute(1, 0)
    inc = torch.arange(8) * 1024
    inc = inc.to(device)
    predicted_idx = (predicted_sequences + inc.reshape(-1, 1))
    predicted_zq = audiodec.rx_encoder.lookup(predicted_idx)
    y = audiodec.decoder.decode(predicted_zq)[:, :, :wav_input.size(-1)]
    y = y.squeeze(1).transpose(1, 0).cpu().numpy()
    sf.write('predicted_audio.wav', y, 24000, "PCM_16")
