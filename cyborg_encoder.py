from transformers import AutoProcessor, Wav2Vec2ConformerModel
import torch
from datasets import load_dataset
import soundfile as SF
from utils.audiodec import AudioDec, assign_model
import  os
import torch
import numpy as np
from tqdm import tqdm
import time
import torchaudio
import soundfile as sf
import shutil

def tokenize_wav(wav_path,audiodec,device,sample_rate=24000):
    wav, sr = torchaudio.load(wav_path)
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

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

# audio file is decoded on the fly
# might need to add batch dimension wax.unsqueeze(0)

inputs = processor(wav, sampling_rate=sampling_rate, return_tensors="pt")
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

prompt_token = tokenize_wav('input.wav',audiodec,device,sample_rate)
print("audio prompt",prompt_token)
print('audio prompt shape',prompt_token.shape)
print("audio promt len",len(prompt_token[0]))


for time_slice in range(len(prompt_token)):
    print(prompt_token[time_slice])


# next we need to align promt_token and last_hidden_states to same length

# using something like promt_token // len(last_hidden_states) , and add last_hidden_states[slice] , to prompt_token[previous_slice_length_end:previous_slice_length_end+slice_length] , where slice_length = promt_token // len(last_hidden_states)

# apply a length regulator for compression, with something like , pool(prompt_token[:compression ratio])





