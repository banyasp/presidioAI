from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
legal_text = "The parties agree that jurisdiction and venue for any dispute arising out of this agreement shall be in the courts of New York."
inputs = tokenizer(legal_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
token_embeddings = outputs.last_hidden_state
sentence_embedding = token_embeddings[:, 0, :]

print(
    f"Shape of token embeddings (batch_size, sequence_length, hidden_size): {token_embeddings.shape}"
)
print(
    f"Shape of sentence embedding (batch_size, hidden_size): {sentence_embedding.shape}"
)
print(f"Sentence Embdding: {sentence_embedding}")
print("\nLegal-BERT successfully generated the sentence embedding.")
