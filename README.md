# CyborgVoice
This the streamingVoice conversion component 

ASR component: Fast U2++ (CTC + Transformer) for semantic representation extraction, alternative: https://github.com/ufal/whisper_streaming 

Semantic tokens for seamless models : https://huggingface.co/facebook/w2v-bert-2.0

MQTTS: Single layer Transformer to predict different codes at a time step level

AudioDec: https://github.com/facebookresearch/AudioDec for real-time codec  and reconstruction 


Next steps: 
1.  Setup WeNet streaming ASR conformer CTC + transformer with caching for attention + CNN https://wenet.org.cn/wenet/runtime.html
    The best streaming architectures use conformer with caching as the best technique including seamless, we want the smallest fastest implementation of conformer CTC to predict semantic units for StreamVoice  (Target by end of Tuesday) need a working system to give conformer audio outputs with word boundary 
2. Create the cross-embedding input for LLama like causal decoder with audiodec+ wenet output alternation (By end of Wednesday)   
4. Understand the dataset that we plan to use (probably use that Audiodec was trained)
5. Train/infer ASR on it
6. Start coding components phase by phase and look at the generated samples





