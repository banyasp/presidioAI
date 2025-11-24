import sqlite3
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Global variables to hold models and data
# Structure: { model_name: { 'tokenizer': t, 'model': m, 'embeddings': e } }
loaded_models = {}
case_data = []

MODEL_CONFIGS = {
    "legal-bert": "nlpaueb/legal-bert-base-uncased",
    "harvard-bert": "casehold/legalbert",
    "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2"
}

def get_db_connection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'scotus_cases.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_model_resources(model_key):
    global loaded_models, case_data
    
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
        
    if model_key in loaded_models:
        return loaded_models[model_key]
        
    print(f"Loading resources for {model_key}...")
    model_name = MODEL_CONFIGS[model_key]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate embeddings for all cases using this model
    if not case_data:
        initialize_case_data()
        
    print(f"Generating embeddings for {model_key}...")
    embeddings_list = []
    for case in case_data:
        text = case['case_facts'] if case['case_facts'] else case['case_name']
        emb = generate_embedding(text, tokenizer, model)
        embeddings_list.append(emb)
        
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = np.array([])
        
    loaded_models[model_key] = {
        'tokenizer': tokenizer,
        'model': model,
        'embeddings': embeddings
    }
    print(f"Resources loaded for {model_key}.")
    return loaded_models[model_key]

def generate_embedding(text, tokenizer, model):
    # Truncate to 512 tokens
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding (first token)
    # For sentence-transformers, mean pooling is often better, but CLS is standard for BERT
    token_embeddings = outputs.last_hidden_state
    
    # Simple CLS strategy for now to keep it consistent across BERT models
    sentence_embedding = token_embeddings[:, 0, :].numpy()
    
    # For sentence-transformers specifically, mean pooling is usually preferred
    # if "sentence-transformers" in model.config._name_or_path:
    #     attention_mask = inputs['attention_mask']
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    #     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    #     sentence_embedding = (sum_embeddings / sum_mask).numpy()

    return sentence_embedding

def initialize_case_data():
    global case_data
    print("Loading cases from database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases")
    rows = cursor.fetchall()
    conn.close()
    
    case_data = []
    for row in rows:
        case_info = {
            "case_name": row['case_name'],
            "docket_number": row['docket_number'],
            "case_facts": row['case_facts'],
            "decision": row['decision'],
            "date": row['date'],
            "related_cases": row['related_cases']
        }
        case_data.append(case_info)
    print(f"Loaded {len(case_data)} cases.")

def find_similar_cases(query_text, model_key="legal-bert", top_k=3):
    global case_data
    
    resources = load_model_resources(model_key)
    tokenizer = resources['tokenizer']
    model = resources['model']
    case_embeddings = resources['embeddings']
    
    if len(case_data) == 0:
        return []
        
    query_emb = generate_embedding(query_text, tokenizer, model)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_emb, case_embeddings)
    
    # Get top k indices
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[0][idx]
        case = case_data[idx]
        
        result_item = {
            "case_name": case['case_name'],
            "similarity_score": float(score),
            "case_facts": {
                "Database Source": case['case_facts']
            },
            "judgement": {
                "Database Source": case['decision']
            },
            "related_cases": case['related_cases']
        }
        results.append(result_item)
        
    return results
