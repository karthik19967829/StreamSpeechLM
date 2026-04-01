# StreamSpeechLM

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
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n cyborg-env python=3.11
conda activate cyborg-env
pip install -r requirements.txt
wget https://github.com/facebookresearch/AudioDec/releases/download/pretrain_models_v02/exp.zip
apt-get update
apt-get install unzip
unzip exp.zip
#get data using the following
git clone https://github.com/nii-yamagishilab/VCC2020-database/blob/master/vcc2020_task_explanation.txt
```
## Run Training/Inference 
python3.11 train_one_step.py 

## Progress tracker 

Jun 14th 
1. Added audio tokenizer function from vall-e Audiodec
2. Updated Cyborg Encoder

Next steps:
1. Embed audio tokens 8 to Embeddings concatenated and projected to a single dimension (de-risked)
2. Project both ASR embedding and audio token embedding to same dimension (de-risked)
3. Length regulation / combining both Embed tokens and ASR embedings to create cross embedding correctly is the key (try to find reference for length regulation / cross embedding) 

June 16th 
1. Length interpolation based on linear component of this https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html added to ASR upsample and align to shape of audio discrete tokens 
2. Added setup.sh to make setup easier 


Next steps:
1. Embed audio tokens 8 to Embeddings concatenated and projected to a single dimension (de-risked)
2. Project both ASR embedding and audio token embedding to same dimension (de-risked)

June 17th 
1. Aligning bottle neck regulator with length interpolation  
2. Added llama lm
3. Acoustic predictor

Next steps:
1. Add training workflow , train on VCC2020 Task2

June 18th 
1. Added VCC2020 dataset exploration 
2. Added rough GPT-4 based refactored code , do more thorough testing and then merge 

Next steps:
1. Add training workflow, train on VCC2020 Task2
2. The source and target length are not exactly same (we might have to set the target length and make source length, or fix the source length and make target of the same shape) , what ever is the strategy is the decoded idea before finalizing
3. Add data loader
4. Make everything a torch nn.module, prepare data accordingly
5. Generate synthetic audio data using elevanlabs API
6. Train one iteration model , do one full cycle and then iterate from there 
   
June 18th 
1. Added code reference for eleven labs data distillation

Next Steps:
1. Create Nikhil's voice on eleven labs
2. Run inference on all Male utterances of VCC2020 Task 1/2
3. Create a dataset in CSV / write output in files _ Nikhil
4. Verify AudioDec length if its the same , then proceed with the training code 

June 20th 
1. Added forward pass for train/inference for CyborgEncoder (ASR/Audiodec/LLama/Acoustic Predictor)

Next Steps:
1. fix the code for correctness / optimize linear algebra with GPT-4
2. add cross entropy loss and do one step forward / backward pass
3. create a simple data-loader that gets input/output wav from VCC2020 Task 1/2 (or from LibriTTS , better to keep it to something Audiodec was trained on)  
4. Add Pad ID to the embedding of llama model (just have a PAD tensor) , if padding is needed during batching
5. Do one iteration of train/eval the pipeline
6. Iterate above till you get some model , thats working as a baseline
7. Add bottle neck regulator / Teacher forcing loss for future prediction (this will make edge predictions easier)  
8. Create Nikhil's voice on eleven labs 
9. Run inference on all Male utterances of VCC2020 Task 1/2
10. Create a dataset in CSV / write output in files _ Nikhil

Jun 22nd 
1. Added Teacher loss
2. Added cross-entropy loss
3. Added training loop
4. Added save functionality 

Notes: Model is now 2.3 GB because of 2.2 GB ASR Model 

Next steps:
1. Load this model and do inference on the overfitted example, just to ensure decoding works
2. Prepare the dataset and do a larger scale training to overfit on VCC2020
3. Inference and check for different source speakers
4. De-risk inference by using a smaller streaming based ASR model for smaller size
5. figure out how to run inference efficiently with some simulated streaming and benchmark the RTF / latency
6. Export with ONNX / other techniques to speed up further
7. Add Multi processing across timesteps for AR predictor 
8. scale training data / add elevan labs based synthetic dataset for Nikhil/ a target speaker 
9. Add pytorch based distributed training to train across Multiple GPUs when needed 
12. DO continued training of AudioDec on the above  

