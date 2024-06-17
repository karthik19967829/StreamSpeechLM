# CyborgVoice
This the streamingVoice conversion component 

ASR component: Fast U2++ (CTC + Transformer) for semantic representation extraction, alternative: https://github.com/ufal/whisper_streaming 

Semantic tokens for seamless models : https://huggingface.co/facebook/w2v-bert-2.0

MQTTS: Single layer Transformer to predict different codes at a time step level

AudioDec: https://github.com/facebookresearch/AudioDec for real-time codec  and reconstruction 


Next steps: 
1.  Get the Conformer encoder / Audio dec token embeddings, alternatively to create cross embeddings 
2.  create a 4 LLama causal model
3.  create a sub decoder to take the output of 4 layers causal model , at a timestamp and predict the 4 Audiodec tokens
4.  pass the predicted Audiodec tokens to the Audiodec decoder and generate audio
5.  setup the training pipeline for the llama decoder and sub-decoder
6.  debug / eval iterate
7.  replace the ASR encoder with chunk chunk-wise conformer
8.  repeat training
9.  add future prediction training guided by teacher, semantic token masking
10.  optimize and check RTF on CPU
11.  Optimize for windows runtime
12.  CyborgVoice ready! 



setup 

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda-latest-Linux-x86_64.sh
./Miniconda-latest-Linux-x86_64.sh
conda create -n cyborg-env python=3.11
pip install -r requirements.txt
```

## Progress tracker 

Jun 14th 
1. Added audio tokenizer function from vall-e Audiodec
2. Updated Cyborg Encoder

Next steps:
1. Embed audio tokens 8 to Embeddings concatenated and projected to a single dimension (de-risked)
2. Project both ASR embedding and audio token embedding to same dimension (de-risked)
3. Length regulation / combining both Embed tokens and ASR embedings to create cross embedding correctly is the key (try to find reference for length regulation / cross embedding)
