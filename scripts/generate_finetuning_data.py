from openai import OpenAI
from prompts.gpt4.dataset_construction_prompt import get_prompt
import os
from dotenv import load_dotenv
import json
import pandas as pd
from tqdm.auto import tqdm
import random
import traceback
load_dotenv()


def set_random_seed(seed):
    print(f'Setting random seed {seed}')
    random.seed(seed)


def generate_unique_ints(n, limit):
    if n > limit:
        raise ValueError("n cannot be greater than limit")
    return random.sample(range(1, limit + 1), n)


def get_client(key=None):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=key or os.environ.get("OPENAI_API_KEY"),
    )
    return client


def generate_response(client: OpenAI, prompt, model="gpt-4", system_prompt: str ="You are a helpful assistant."):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,  # You can adjust this value as needed
            # temperature=0.7   # Controls randomness (higher = more random, lower = more focused)
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    client = get_client()
    seed = 0
    set_random_seed(seed)
    with open('/Users/soham/Desktop/CyborgVoice/data/finetuning_data/topics.json') as f:
        topics = json.load(f)['topics']

    outfile = "generated_gpt4_data.csv"
    if os.path.isfile(outfile):
        data = pd.read_csv(outfile, index_col=0)
    else:
        data = pd.DataFrame(columns=['topic', 'kwargs', 'prompt', 'response'])
        
    index = -1
    try:
        for topic in tqdm(topics[:10]):
            index += 1
            if index < len(data) and data.loc[index, 'topic'] == topic and len(data.loc[index, 'response']) > 0:
                continue
            num_rounds = random.randint(10, 21)
            response_word_count = random.randint(10, 21)
            interrupted_response_word_count = random.randint(10, 21)
            round_nums = generate_unique_ints(8, num_rounds)

            kwargs = dict(
                num_rounds=num_rounds,
                denial_round=round_nums[0],
                inquiry_round=round_nums[1],
                topic_change_round=round_nums[2],
                noise_round=round_nums[3],
                acknowledgment_round=round_nums[4],
                lack_round=round_nums[4],
                complete_round=round_nums[6],
                error_round=round_nums[7],
                first_question_topic=topic,
                response_word_count=response_word_count,
                interrupted_response_word_count=interrupted_response_word_count
            )
            prompt = get_prompt(**kwargs)

            response = generate_response(client, prompt)
            data.loc[index] = [topic, kwargs, prompt, response]
            data.to_csv(outfile)
    except:
        traceback.print_exc()


if __name__ == "__main__":
    main()
