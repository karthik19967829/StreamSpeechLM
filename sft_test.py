import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm.auto import tqdm

# Define a simple dataset class
class SimpleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Prepare data (you would replace this with your actual data loading and preprocessing logic)
input_ids = [
    [0, 901, 250, 315, 30, 677, 900, 615, 643, 1, 901, 250, 189, 88, 677, 942, 833, 482, 1, 2],
    [0, 901, 250, 60, 88, 677, 820, 280, 319, 1, 901, 250, 60, 88, 677, 820, 795, 319, 1, 2]
]  # Example input IDs
labels = [0, 1]  # Example labels for some task

# Tokenizer and model loading
#tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Prepare dataset
dataset = SimpleDataset({'input_ids': input_ids, 'labels': labels})
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # number of epochs
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

print("Training completed.")
