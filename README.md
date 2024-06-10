# CyborgVoice
This the streamingVoice conversion component 

ASR component: Fast U2++ (CTC + Transformer) for semantic representation extraction, alternative: https://github.com/ufal/whisper_streaming 

Semantic tokens for seamless models : https://huggingface.co/facebook/w2v-bert-2.0

MQTTS: Single layer Transformer to predict different codes at a time step level

AudioDec: https://github.com/facebookresearch/AudioDec for real-time codec  and reconstruction 


Next steps: 
1. Understand the dataset that we plan to use (probably use that audiodec was trained)
2. Train / infer ASR on it
3. Start coding components phase by phase and look at the generated samples





