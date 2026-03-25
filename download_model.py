import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_model():
    model_name = "google/flan-t5-base"
    cache_dir = "./flan-t5-base"
    if not os.path.exists(cache_dir):
        print(f"Downloading {model_name} to {cache_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer.save_pretrained(cache_dir)
        model.save_pretrained(cache_dir)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
