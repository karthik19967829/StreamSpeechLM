from transformers import LlamaModel, LlamaConfig
import torch 
configuration = LlamaConfig()

configuration.hidden_size = 1024
configuration.num_hidden_layers = 6
configuration.num_attention_heads = 8
configuration.num_key_value_heads = 8
configuration.intermediate_size= 4096


print("configuration",configuration)
model = LlamaModel(configuration)
print(model)
# Example test input
test_input = torch.randn(1, 10, 1024)  # [batch size, sequence length, hidden size]
test_position_ids = torch.arange(0, 10).unsqueeze(0)  # Position IDs for 10 positions

#, position_ids=test_position_ids
last_hidden_state = test_input

for layer in model.layers:
    try:
        last_hidden_state = layer(last_hidden_state,position_ids=test_position_ids)[0]
        print("Output shape from single layer:", last_hidden_state.shape)
    except Exception as e:
        print("Error during test layer pass:", e)

from transformers import LlamaModel, LlamaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressivePredictor(nn.Module):
    def __init__(self, config):
        super(AutoregressivePredictor, self).__init__()
        self.acoustic_predictor_model = LlamaModel(config)
        self.embedding = nn.Embedding(1024, 256)
        self.output_layer = nn.Linear(config.hidden_size, 1024)

    def forward(self, initial_hidden_state , inference=True):
        current_input = initial_hidden_state
        predicted_tokens = []
        
        if inference:
            for position in range(8):  # Generate 8 tokens autoregressively
                # Forward pass through the model
                outputs = self.acoustic_predictor_model.layers[0](current_input,position_ids = torch.arange(0, current_input.shape[1]).unsqueeze(0))
                print("outputs shape",outputs[0].shape)
                #logits = outputs[0].last_hidden_state[:, -1, :]
                transformed_logits = self.output_layer(outputs[0][:, -1, :])
                print("transformed logit",transformed_logits.shape)
                probabilities = F.softmax(transformed_logits, dim=-1)
                predicted_token = torch.argmax(probabilities, dim=-1)
                predicted_tokens.append(predicted_token)
                #logits = outputs[0].logits  # Shape: [batch_size, seq_length, vocab_size]
                # Prepare the input for the next step
                token_embedding = self.embedding(predicted_token)
                print("current input shape",current_input.shape,"token embedding shape",token_embedding.shape)
                token_embedding = token_embedding.unsqueeze(0)
                current_input = torch.cat([current_input, token_embedding], dim=1)
        #print("predicted tokens",predicted_tokens)
        return predicted_tokens  # Shape: [batch_size, 8]

# Configuration for the acoustic predictor
acoustic_configuration = LlamaConfig()

acoustic_configuration.hidden_size = 256
acoustic_configuration.num_hidden_layers = 1
acoustic_configuration.num_attention_heads = 4
acoustic_configuration.num_key_value_heads = 4
acoustic_configuration.intermediate_size= 1024

# Initialize the acoustic predictor module
ar_predictor = AutoregressivePredictor(acoustic_configuration)

# Example initial hidden state
initial_hidden_state = torch.randn(1, 1, 256)  # [batch_size, 1 (initial sequence length), hidden_size]

# Perform autoregressive prediction
predicted_sequence = ar_predictor(initial_hidden_state)
print("Predicted sequence of tokens:", predicted_sequence)
