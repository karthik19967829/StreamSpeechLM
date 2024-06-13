from transformers import AutoProcessor, Wav2Vec2ConformerModel
import torch
from datasets import load_dataset
import soundfile as sf
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

data, fs = sf.read("/workspace/CyborgVoice/output.wav", always_2d=True)
# audio file is decoded on the fly
#
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape))

import os
import torch
import argparse
import numpy as np
from utils.audiodec import AudioDec, assign_model


sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model("libritts_v1")
print("AudioDec initinalizing!")
tx_device = f'cuda:0'
rx_device = f'cuda:0'
audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
with torch.no_grad():
    #data = dataset[0]["audio"]["array"]
    #print(data,data.shape)
    x = np.expand_dims(data.transpose(1, 0), axis=1) # (T, C) -> (C, 1, T)
    x = torch.tensor(x, dtype=torch.float).to(tx_device)
    print("Encode/Decode...")
    z = audiodec.tx_encoder.encode(x)
    idx = audiodec.tx_encoder.quantize(z)
    print("index",idx)
    zq = audiodec.rx_encoder.lookup(idx)
    print("zq",zq)
    y = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
    y = y.squeeze(1).transpose(1, 0).cpu().numpy() # T x C
    sf.write(
        'test_output.wav',
        y,
        fs,
        "PCM_16",
    )
    print(f"Output test_output.wav!")






