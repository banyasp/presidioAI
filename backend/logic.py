import sqlite3
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Global variables to hold model and data
tokenizer = None
model = None
case_data = []
case_embeddings = None

def get_db_connection():
    # Assumes the script is run from the project root or backend folder
    # We'll try to find the DB relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'scotus_cases.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_model():
    global tokenizer, model
    print("Loading Legal-BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
    print("Model loaded.")

def get_embedding(text):
    global tokenizer, model
    if tokenizer is None or model is None:
        load_model()
    
    # Truncate to 512 tokens as BERT has a limit
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding (first token) as the sentence embedding
    # Alternatively, we could mean pool the last hidden state
    token_embeddings = outputs.last_hidden_state
    sentence_embedding = token_embeddings[:, 0, :].numpy()
    return sentence_embedding

def initialize_data():
    global case_data, case_embeddings
    print("Loading cases from database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cases")
    rows = cursor.fetchall()
    conn.close()
    
    case_data = []
    embeddings_list = []
    
    print(f"Found {len(rows)} cases. Generating embeddings (this may take a moment)...")
    
    for row in rows:
        # Combine case facts and decision for a rich representation, or just use facts
        # Using case_facts for similarity search seems most appropriate for "finding similar cases" based on input facts
        text_to_embed = row['case_facts'] if row['case_facts'] else ""
        if not text_to_embed:
            text_to_embed = row['case_name'] # Fallback
            
        emb = get_embedding(text_to_embed)
        
        case_info = {
            "case_name": row['case_name'],
            "docket_number": row['docket_number'],
            "case_facts": row['case_facts'],
            "decision": row['decision'],
            "date": row['date']
        }
        case_data.append(case_info)
        embeddings_list.append(emb)
        
    if embeddings_list:
        case_embeddings = np.vstack(embeddings_list)
    else:
        case_embeddings = np.array([])
        
    print("Initialization complete.")

def find_similar_cases(query_text, top_k=3):
    global case_embeddings, case_data
    
    if not case_data:
        initialize_data()
        
    if len(case_data) == 0:
        return []
        
    query_emb = get_embedding(query_text)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_emb, case_embeddings)
    
    # Get top k indices
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[0][idx]
        case = case_data[idx]
        
        # Structure the output as requested:
        # "output sections like case facts and judgement and then withing them wil lbe names of different model that gave output"
        # Since we only have the DB, we'll use "Database Source" as the primary model
        # and generate a "Mock Model" for demonstration.
        
        result_item = {
            "case_name": case['case_name'],
            "similarity_score": float(score),
            "case_facts": {
                "Database Source": case['case_facts']
            },
            "judgement": {
                "Database Source": case['decision']
            }
        }
        results.append(result_item)
        
    return results
