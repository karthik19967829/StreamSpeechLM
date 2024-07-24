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

wav_path = 'input.wav'
asr_sample_rate = 16000
wav, sr = torchaudio.load(wav_path) #C T-> 1 C T
print("wav shape",wav.shape)  

model_name = "libritts_v1"
device = 'cpu' 
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device , rx_device=device )
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

prompt_token = tokenize_wav(wav_path,audiodec,device,sample_rate)
print(type(prompt_token[:4]))


def prepare_for_llama(tokens, bos_token_id, sep_token_id, eos_token_id):
    """
    Prepare the tokens for LLaMA model by adding BOS, SEP, and EOS tokens.

    Args:
    tokens (np.ndarray): Array of tokens from the tokenizer.
    bos_token_id (int): Beginning of sentence token ID.
    sep_token_id (int): Separator token ID.
    eos_token_id (int): End of sentence token ID.

    Returns:
    list: A list of token IDs ready to be input into LLaMA.
    """
    # Flatten the token array
    token_list = tokens.flatten()

    # Create a new list starting with the BOS token
    input_ids = [bos_token_id]

    # Append tokens with SEP after every 8 tokens
    for i in range(0, len(token_list), 8):
        input_ids.extend(token_list[i:i+8])
        input_ids.append(sep_token_id)

    # Replace the last SEP token with an EOS token
    input_ids[-1] = eos_token_id

    return input_ids


# Define the special token IDs (these should be defined according to your tokenizer)
BOS_TOKEN_ID = 0
SEP_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# Prepare the tokens for input to LLaMA
input_ids = prepare_for_llama(prompt_token, BOS_TOKEN_ID, SEP_TOKEN_ID, EOS_TOKEN_ID)
print(input_ids)
