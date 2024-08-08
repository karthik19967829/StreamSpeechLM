from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchaudio
import numpy as np
from utils.audiodec import AudioDec, assign_model

# Load the trained model and tokenizer
model_checkpoint = "results/checkpoint-340"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16).to('cuda')

if '<sosp>' not in tokenizer.get_vocab():
    units_size=1024
    #logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
    new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
    tokenizer.add_tokens(new_tokens)

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

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

def create_sliding_windows(tokens, window_size, step_size):
    windows = []
    for start in range(0, len(tokens) - window_size + 1, step_size):
        end = start + window_size
        window = tokens[start:end]
        if len(window) == window_size:
            windows.append("".join(window))
    return windows

# Function to generate target speech from source speech
def generate_target_speech(source_window):
    system_message = "using the speech prompt (a sequence of speech frames, with each frame represented by 8 audio codes, which have values with a range [0-1023]), given a source speaker sequence of the same shape, generate corresponding target speaker speech of the same speech"
    user_message = source_window
    prompt_input_ids = tokenizer.apply_chat_template(
        conversation=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
        add_generation_prompt=True,
        tokenize=True,
        add_special_tokens=True
    )
    input_ids = torch.tensor(prompt_input_ids).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=len(input_ids[0]) + 80, eos_token_id=tokenizer.eos_token_id)
    
    generated_ids = output[0].cpu().numpy()
    return generated_ids[len(input_ids[0]):].tolist()

# Load the AudioDec model
model_name = "libritts_v1"
device = 'cuda'
sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model_name)
audiodec = AudioDec(tx_device=device, rx_device=device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

# Example: Load source and target windows from audio files
source_file_wav_path = '/workspace/VCC2020-database/target_task1/TEF2/E10051.wav'  # Replace with the actual path to the source audio file
target_file_wav_path = '/workspace/VCC2020-database/target_task1/TEF1/E10051.wav'  # Replace with the actual path to the target audio file

source_prompt_token = tokenize_wav(source_file_wav_path, audiodec, device, sample_rate)
target_prompt_token = tokenize_wav(target_file_wav_path, audiodec, device, sample_rate)

source_windows = create_sliding_windows(source_prompt_token, window_size=10, step_size=5)
target_windows = create_sliding_windows(target_prompt_token, window_size=10, step_size=5)

# Generate target speech and check
for source_window, target_window in zip(source_windows, target_windows):
    generated_target = generate_target_speech(source_window)
    decoded_generated_target = tokenizer.decode(generated_target)
    print("decoded target:",decoded_generated_target)
    is_match = np.array_equal(np.array(generated_target), target_window)
    print("target window:",target_window)
    print(f"Do the generated target and actual target match? {'Yes' if is_match else 'No'}")
