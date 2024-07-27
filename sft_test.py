# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
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


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16)

num_speech_tokens = 1024  # Number of speech tokens
num_special_tokens = 5    # BOS, EOS, SEP, PAD for user and assistant
total_new_tokens = num_speech_tokens + num_special_tokens
model.resize_token_embeddings(total_new_tokens)
if model.model.embed_tokens.weight.dtype != torch.bfloat16:
    model.model.embed_tokens.weight = model.model.embed_tokens.weight.to(torch.bfloat16)

print(model.model.embed_tokens.weight.shape)
# Update the model configuration to reflect the new vocabulary size
#model.config.vocab_size = total_new_tokens
sep_token_id = 1024
prompt_bos = 1025
prompt_eos = 1026
assistant_bos = 1027
assistant_eos = 1028
# Define your input_ids and labels


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

model_name = "libritts_v1"
device = 'cpu' 
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device , rx_device=device )
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

file_list = os.listdir('/workspace/VCC2020-database/target_task1/TEF2')
print(file_list)

#def get_prompt(prompt_tensor):
def flatten_output_tensor(tensor, sep_token_id=1024, assistant_bos=1027, assistant_eos=1028):
    # Initialize the sequence with the assistant_bos token
    sequence = [assistant_bos]
    
    # Loop through each row in the tensor
    for row in tensor:
        # Add the current row (8 tokens)
        sequence.extend(row.tolist())
        # Add the separator token
        sequence.append(sep_token_id)
    
    # Replace the last sep_token_id with assistant_eos
    sequence[-1] = assistant_eos
    
    # Convert the sequence list back to a tensor
    return torch.tensor(sequence)

def flatten_tensor_with_tokens(tensor, sep_token_id=1024, prompt_bos=1025, prompt_eos=1026):
    # Initialize the sequence with the prompt_bos token
    sequence = [prompt_bos]
    
    # Loop through each row in the tensor
    for row in tensor:
        # Add the current row (8 tokens)
        sequence.extend(row.tolist())
        # Add the separator token
        sequence.append(sep_token_id)
    
    # Replace the last sep_token_id with prompt_eos
    sequence[-1] = prompt_eos
    
    # Convert the sequence list back to a tensor
    return torch.tensor(sequence)

def prepare_llm_data(input_tensor, output_tensor, ignore_index=-100,
                     prompt_eos=1026, assistant_bos=1027):
    # Flatten the input tensor with special tokens
    flattened_input = flatten_tensor_with_tokens(input_tensor)
    
    # Flatten the output tensor with different special tokens
    flattened_output = flatten_output_tensor(output_tensor)
    
    # Construct input_ids by concatenating input tensor with all but the last token of the output tensor
    input_ids = torch.cat((flattened_input, flattened_output[:-1]))

    # Create labels: ignore index for all of the input part, and then use the output tensor from the second token
    labels = torch.cat((torch.full_like(flattened_input, ignore_index), flattened_output[1:]))

    return input_ids, labels



input_ids_list = []
labels_list = []

# Loop over each file in the file list
for file in file_list:
    source_file_wav_path = f'/workspace/VCC2020-database/target_task1/TEF2/{file}'
    target_file_wav_path = f'/workspace/VCC2020-database/target_task1/TEF1/{file}'
    
    # Tokenize source and target audio files
    source_prompt_token = tokenize_wav(source_file_wav_path, audiodec, device, sample_rate)[:141]
    target_prompt_token = tokenize_wav(target_file_wav_path, audiodec, device, sample_rate)[:141]
    
    # Prepare input IDs and labels
    input_ids, labels = prepare_llm_data(source_prompt_token, target_prompt_token, ignore_index=-100, prompt_eos=1026, assistant_bos=1027)
    
    # Append to lists
    input_ids_list.append(input_ids.tolist())
    labels_list.append(labels.tolist())

# Create the dataset
data = {'input_ids': input_ids_list, 'labels': labels_list}
dataset = Dataset.from_dict(data)


'''input_ids = [
    [0, 901, 250, 315, 30, 677, 900, 615, 643, 1, 901, 250, 189, 88, 677, 942, 833, 482, 1, 2],
    [0, 901, 250, 60, 88, 677, 820, 280, 319, 1, 901, 250, 60, 88, 677, 820, 795, 319, 1, 2]
]
labels = [
    [0, 901, 250, 315, 30, 677, 900, 615, 643, 1, 901, 250, 189, 88, 677, 942, 833, 482, 2, 2],
    [0, 901, 250, 60, 88, 677, 820, 280, 319, 1, 901, 250, 60, 88, 677, 820, 795, 319, 2, 2]
]

# Create a dataset
data = {'input_ids': input_ids, 'labels': labels}
dataset = Dataset.from_dict(data)'''

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=100,              # total number of training epochs
    per_device_train_batch_size=32,   # batch size per device during training
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Define a data collator
def data_collator(features):
    input_ids = torch.tensor([feature['input_ids'] for feature in features], dtype=torch.long)
    labels = torch.tensor([feature['labels'] for feature in features], dtype=torch.long)
    
    # Set the labels of the prompt tokens to -100
    for label in labels:
        prompt_end = (label == 1).nonzero(as_tuple=True)[0][-1].item() + 1
        label[:prompt_end] = -100
    
    batch = {
        'input_ids': input_ids,
        'labels': labels
    }
    return batch

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,               # training dataset
    data_collator=data_collator ,
    ogging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",  # Evaluate every so many steps
    eval_steps=500,               # Steps at which evaluation occurs
    load_best_model_at_end=True         # data collator for dynamic padding
)

# Train the model
trainer.train()
