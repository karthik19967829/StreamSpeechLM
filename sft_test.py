# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

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

# Define your input_ids and labels
input_ids = [
    [0, 901, 250, 315, 30, 677, 900, 615, 643, 1, 901, 250, 189, 88, 677, 942, 833, 482, 1, 2],
    [0, 901, 250, 60, 88, 677, 820, 280, 319, 1, 901, 250, 60, 88, 677, 820, 795, 319, 1, 2]
]
labels = [
    [0, 901, 250, 315, 30, 677, 900, 615, 643, 1, 901, 250, 189, 88, 677, 942, 833, 482, 2, 2],
    [0, 901, 250, 60, 88, 677, 820, 280, 319, 1, 901, 250, 60, 88, 677, 820, 795, 319, 2, 2]
]

# Create a dataset
data = {'input_ids': input_ids, 'labels': labels}
dataset = Dataset.from_dict(data)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=100,              # total number of training epochs
    per_device_train_batch_size=2,   # batch size per device during training
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
    data_collator=data_collator          # data collator for dynamic padding
)

# Train the model
trainer.train()
