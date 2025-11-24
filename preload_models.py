from transformers import AutoTokenizer, AutoModel

MODEL_CONFIGS = {
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    "harvard-bert": "casehold/legalbert",
    "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2"
}

def preload():
    for key, name in MODEL_CONFIGS.items():
        print(f"Downloading {key}: {name}...")
        try:
            AutoTokenizer.from_pretrained(name)
            AutoModel.from_pretrained(name)
            print(f"Successfully downloaded {key}")
        except Exception as e:
            print(f"Failed to download {key}: {e}")

if __name__ == "__main__":
    preload()
