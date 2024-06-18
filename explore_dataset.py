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

model_name = "libritts_v1"
device = 'cpu' 
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device , rx_device=device )
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

source_wav_path = '/workspace/VCC2020-database/source/SEF1/E10053.wav'
target_wav_path = '/workspace/VCC2020-database/target_task1/TEF1/E10053.wav'
source_prompt_token = tokenize_wav(source_wav_path,audiodec,device,sample_rate)
target_prompt_token = tokenize_wav(target_wav_path,audiodec,device,sample_rate)
print("source length",len(source_prompt_token))
print("target length",len(target_prompt_token))


