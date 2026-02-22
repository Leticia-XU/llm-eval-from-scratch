import requests
import re
from datasets import load_dataset
from tqdm import tqdm

API_URL = "http://localhost:7777/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct-AWQ"

def extract_answer(text):
    match = re.search(r"#### (\-?\d+)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"\-?\d+", text)
    return numbers[-1] if numbers else None

def ask_model(question):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a math reasoning assistant. Give the final answer in the format: #### number"},
            {"role": "user", "content": question}
        ],
        "temperature": 0.0
    }
    response = requests.post(API_URL, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def main():
    dataset = load_dataset("gsm8k", "main", split="test[:50]")
    correct = 0

    for example in tqdm(dataset):
        question = example["question"]
        gold_answer = extract_answer(example["answer"])

        model_output = ask_model(question)
        pred_answer = extract_answer(model_output)

        if pred_answer == gold_answer:
            correct += 1

    print("Accuracy:", correct / len(dataset))

if __name__ == "__main__":
    main()
