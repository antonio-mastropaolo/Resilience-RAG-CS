import openai
import pandas as pd
from tqdm import tqdm



client = openai.OpenAI(
    api_key='' #ADD key here
)

SEED = 123


def get_chat_response(prompt: str, seed: int = SEED, temperature: float = 0.0):
    try:
        messages = [
            {"role": "system", "content": "You are a Java developer and your task is to summarize Java methods"},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            seed=seed,
            max_tokens=100,
            temperature=temperature,
        )

        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        #print(f"An error occurred: {e}")
        return "<NONE>"


def main():
    dataset = pd.read_csv('baseline.csv')
    responses = []
    for idx, row in tqdm(dataset.iterrows()):

        gptResult = get_chat_response(row['prompt']).strip()
        responses.append(gptResult.strip())

        print(f"[{idx}]: {gptResult}")
        with open('predictions@baseline.txt','a+') as fwrite:
            fwrite.write(f"{idx}: {gptResult.strip()}\n")


    dataset['gpt4-raw-predictions'] = responses
    dataset.to_csv('baseline-GPT4o.csv')


if __name__ == "__main__":
    main()
