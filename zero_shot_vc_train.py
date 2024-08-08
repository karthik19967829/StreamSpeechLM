from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from utils.audiodec import AudioDec, assign_model
import os
import torchaudio
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained("/workspace/CyborgVoice/results/checkpoint-20", torch_dtype=torch.bfloat16)

if '<sosp>' not in tokenizer.get_vocab():
    units_size=1024
    logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
    new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
    tokenizer.add_tokens(new_tokens)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))


logger.info("only update embedding parameters")
'''for name, param in model.named_parameters():
    if "embed" not in name:
        param.requires_grad = False'''


# Function to tokenize WAV file
def tokenize_wav(wav_path, audiodec, device, sample_rate=24000):
    wav, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    with torch.no_grad():
        wav = wav.unsqueeze(1)  # C T-> 1 C T
        wav = wav.float().to(device)
        z = audiodec.tx_encoder.encode(wav)
        idx = audiodec.tx_encoder.quantize(z)
        inc = torch.arange(8) * 1024
        idx = idx.cpu() - inc.reshape(-1, 1)
        raw_audio_tokens = idx.numpy().T
        special_audio_tokens_list = []
        for window in raw_audio_tokens:
            special_audio_tokens = '<sosp>'  
            #print(window)
            for audio_token in window:
                special_audio_tokens = special_audio_tokens + f"<{audio_token}>"
            special_audio_tokens = special_audio_tokens + '<eosp>'
            #print(special_audio_tokens)  
            special_audio_tokens_list.append(special_audio_tokens)
        return special_audio_tokens_list  

model_name = "libritts_v1"
device = 'cuda'
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

file_list = os.listdir('/workspace/VCC2020-database/target_task1/TEF2')

def create_sliding_windows(tokens, window_size, step_size):
    windows = []
    for start in range(0, len(tokens) - window_size + 1, step_size):
        end = start + window_size
        window = tokens[start:end]
        if len(window) == window_size:
            windows.append("".join(window))
    return windows

from tqdm import tqdm
# Loop over each file in the file list
input_ids_list = []
labels_list = []
target_spk_prompt_token = tokenize_wav('/workspace/VCC2020-database/target_task1/TEF1/E10051.wav', audiodec, device, sample_rate)
for file in tqdm(file_list):
    source_file_wav_path = f'/workspace/VCC2020-database/target_task1/TEF2/{file}'
    target_file_wav_path = f'/workspace/VCC2020-database/target_task1/TEF1/{file}'
    
    source_prompt_token = tokenize_wav(source_file_wav_path, audiodec, device, sample_rate)
    target_prompt_token = tokenize_wav(target_file_wav_path, audiodec, device, sample_rate)
    
    
    source_windows = create_sliding_windows(source_prompt_token, window_size=10, step_size=5)
    target_windows = create_sliding_windows(target_prompt_token, window_size=10, step_size=5)
    
    for source_window, target_window in zip(source_windows, target_windows):
        print(source_window, target_window)
        system_message = f"using the speech prompt a sequence of speech frames, with each frame represented by 8 audio codes, which have values with a range [0-1023]) given a source speaker sequence of the same shape, generate corresponding target speaker speech of the same speech"
        user_message = source_window
        assistant_message = target_window
        prompt_input_ids = tokenizer.apply_chat_template(conversation=[{"role":"system","content":system_message}, {"role":"user","content":user_message}],add_generation_prompt=True,tokenize=True,add_special_tokens=True)
        input_ids = tokenizer.apply_chat_template(conversation=[{"role":"system","content":system_message}, {"role":"user","content":user_message},{"role":"assistant","content":assistant_message}],tokenize=True,add_special_tokens=True)
        prompt_len = len(prompt_input_ids) 
        output_ids = [-100]*(prompt_len-1) + input_ids[prompt_len:] #shift right by 1 
        input_ids = input_ids[:-1] #remove the last EOS for the input
        assert len(input_ids)==len(output_ids),"there is length mismatch"
        input_ids_list.append(input_ids)
        labels_list.append(output_ids)

input_lens = set([len(input_ids) for input_ids in input_ids_list])
output_lens = set([len(output_ids) for output_ids in labels_list])

print("====input lens=====",input_lens,output_lens)
print("num data samples==== ",len(input_ids_list))


train_data = {'input_ids': input_ids_list[:-1000], 'labels': labels_list[:-1000]}
train_dataset = Dataset.from_dict(train_data)

eval_data = {'input_ids': input_ids_list[-1000:], 'labels': labels_list[-1000:]}
eval_dataset = Dataset.from_dict(eval_data)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="steps",
    weight_decay=0.01,
    learning_rate=2e-5,
    gradient_accumulation_steps=128,
    save_steps=128,
    evaluation_strategy="steps",
    eval_steps=64,
    load_best_model_at_end=True
)

def data_collator(features):
    #print(f"input_ids: {[feature['input_ids'] for feature in features]}")
    input_ids = torch.tensor([feature['input_ids'] for feature in features], dtype=torch.long)
    labels = torch.tensor([feature['labels'] for feature in features], dtype=torch.long)
    batch = {'input_ids': input_ids, 'labels': labels}
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()

