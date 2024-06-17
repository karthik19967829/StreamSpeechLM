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
print('audio prompt shape',prompt_token.shape)

#align ASR output to shape of prompt audio tokens
last_hidden_states_permuted = last_hidden_states.permute(0, 2, 1)
upsampled_hidden_states_tensor_permuted = F.interpolate(last_hidden_states_permuted, size=len(prompt_token), mode='linear',align_corners=True)
upsampled_hidden_states_tensor_permuted = upsampled_hidden_states_tensor_permuted.permute(0, 2, 1)
print("interpolated last hidden state shape",upsampled_hidden_states_tensor_permuted.shape)







