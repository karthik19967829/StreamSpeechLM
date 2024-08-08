import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ConformerModel, LlamaModel, LlamaConfig
from utils.audiodec import AudioDec, assign_model

# Load model and processor
def load_model_processor():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
    model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
    return processor, model

# Preprocess the WAV file
def preprocess_wav(wav_path, target_sample_rate=16000):
    wav, sr = torchaudio.load(wav_path)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav.squeeze(0), target_sample_rate

# Tokenize WAV for model input
def tokenize_wav(wav_path, audiodec, device,sample_rate=24000):
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

# ASR processing
def process_asr(wav_path, device='cpu'):
    processor, model = load_model_processor()
    wav, sr = preprocess_wav(wav_path, target_sample_rate=16000)
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state

# Audio decoding setup
def setup_audio_decoding(device):
    model_name = "libritts_v1"
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
    audiodec = AudioDec(tx_device=device, rx_device=device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)
    return audiodec, sample_rate

# Autoregressive decoder
class AutoregressiveDecoder(nn.Module):
    def __init__(self, config):
        super(AutoregressiveDecoder, self).__init__()
        self.acoustic_predictor_model = LlamaModel(config)
        self.embedding = nn.Embedding(1024, 256)
        self.output_layer = nn.Linear(config.hidden_size, 1024)

    def forward(self, initial_hidden_state,inference=True):
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

# Main function
def main():
    device = 'cpu'
    wav_path = 'input.wav'
    hidden_states = process_asr(wav_path, device=device)
    audiodec, sample_rate = setup_audio_decoding(device=device)
    prompt_token = tokenize_wav(wav_path, audiodec, device, sample_rate)
    print('Audio prompt shape:', prompt_token.shape)

    decoder_config = LlamaConfig(
        hidden_size=256,
        vocab_size=1024,  # Assuming 1024 possible audio tokens
        sequence_length=8,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024
    )
    decoder = AutoregressiveDecoder(decoder_config)
    initial_hidden_state = torch.randn(1, 1, 256)  # Example initial hidden state
    generated_tokens = decoder(initial_hidden_state)
    print("Generated sequence of tokens:", generated_tokens)

if __name__ == "__main__":
    main()
