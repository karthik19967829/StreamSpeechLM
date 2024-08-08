from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset,load_dataset
import torch
from utils.audiodec import AudioDec, assign_model
import os
import torchaudio
import logging
import sys
import soundfile as sf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype="auto")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean")

if '<sosp>' not in tokenizer.get_vocab():
    units_size=1024
    logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
    new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
    tokenizer.add_tokens(new_tokens)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))


logger.info("only update embedding parameters")
for name, param in model.named_parameters():
    if "embed" not in name:
        param.requires_grad = False


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

from tqdm import tqdm
# Loop over each file in the file list
input_ids_list = []
labels_list = []
system_message = "convert the given sequence of speech tokens to text ,strictly directly generated the text contained and nothing else"

for example in ds['validation']:
    audio_array = example['audio']['array']
    audio_path = 'audio_example.wav'
    sf.write(audio_path, audio_array,16000)  # Save the audio file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    special_audio_tokens_list  = tokenize_wav(audio_path, audiodec, device, sample_rate=24000)
    user_message = "".join(special_audio_tokens_list)
    assistant_message = example['text'].lower()
    print(user_message)
    print(assistant_message)
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

max_input_len = max([len(input_ids) for input_ids in input_ids_list])

tokenizer.pad_token_id = tokenizer.eos_token_id

for i in range(len(input_ids_list)):
    input_ids_list[i] += [tokenizer.pad_token_id] * (max_input_len - len(input_ids_list[i]))
    labels_list[i] += [-100] * (max_input_len - len(labels_list[i]))
    assert input_ids_list[i]!=None,"input ids None"
    assert labels_list[i]!=None,"label ids None"

print("====input lens=====",input_lens,output_lens)
print("num data samples==== ",len(input_ids_list))


train_data = {'input_ids': input_ids_list[:-10], 'labels': labels_list[:-10]}
train_dataset = Dataset.from_dict(train_data)

eval_data = {'input_ids': input_ids_list[-10:], 'labels': labels_list[-10:]}
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
    load_best_model_at_end=True,
    deepspeed="./ds_config_zero3.json"
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

