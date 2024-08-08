import numpy as np
import soundfile as sf
from datasets import load_dataset
import torchaudio
import torch
from utils.audiodec import AudioDec, assign_model

# Load the dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")

model_name = "libritts_v1"
device = 'cuda'
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
# Tokenize wav function
def tokenize_wav(wav_path, audiodec, device, sample_rate):
    wav, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.unsqueeze(0).float().to(device)  # Changed from unsqueeze(1) to unsqueeze(0)
    with torch.no_grad():
        z = audiodec.tx_encoder.encode(wav)
        idx = audiodec.tx_encoder.quantize(z)
    inc = torch.arange(8) * 1024
    idx = (idx.cpu() - inc.reshape(-1, 1)).numpy().T
    return idx, wav

# Save and tokenize the audio from the dataset
def process_audio(example, sample_rate=24000):
    audio_array = example['audio']['array']
    audio_path = 'audio_example.wav'
    sf.write(audio_path, audio_array,16000)  # Save the audio file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx, wav = tokenize_wav(audio_path, audiodec, device, sample_rate)

    print(f"Text: {example['text']}")
    print(f"Tokenized indices: {idx}")

    with torch.no_grad():
        predicted_sequences = torch.tensor(idx).T.to(device)
        inc = torch.arange(8).to(device) * 1024
        predicted_idx = (predicted_sequences + inc.reshape(-1, 1))
        predicted_zq = audiodec.rx_encoder.lookup(predicted_idx)
        y = audiodec.decoder.decode(predicted_zq)[:, :, :wav.size(-1)]
        y = y.squeeze(1).transpose(1, 0).cpu().numpy()
        sf.write('predicted_audio.wav', y, 24000, "PCM_16")

    print("Reconstructed audio saved as 'predicted_audio.wav'")

# Iterate over the dataset and process the first example
count = 0
for example in ds['validation']:
    if count==2:
        process_audio(example)
        break
    count+=1