import torch
import numpy as np
import sqlite3
import json
from transformers import AutoModel, AutoTokenizer


class LegalBERTEmbedder:
    
    def __init__(self, device: str = "auto", max_length: int = 512):
        self.model_name = "nlpaueb/legal-bert-base-uncased"
        self.max_length = max_length

        # figure out what device to use
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA (GPU): {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS (Apple Silicon GPU)")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        else:
            self.device = torch.device(device)
            print(f"Using specified device: {device}")

        # lazy load these
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print("Model loaded successfully!")

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        # single text embedding
        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            # grab CLS token
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        embedding = embedding.cpu().numpy()

        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def embed_batch(self, texts: list[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        # batch processing for efficiency
        self._load_model()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # CLS token

            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        # normalize makes cosine similarity easier (just dot product)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    def embed_database(self, db_path: str, text_columns: list[str], embedding_columns: list[str], 
                      table_name: str = "cases", batch_size: int = 32, normalize: bool = True) -> dict:
        # process database columns and store embeddings
        if len(text_columns) != len(embedding_columns):
            raise ValueError("text_columns and embedding_columns must have the same length")

        print(f"\nConnecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        stats = {"total_rows": 0, "columns_processed": {}}

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats["total_rows"] = cursor.fetchone()[0]
            print(f"Total rows in {table_name}: {stats['total_rows']}")

            # add embedding columns if needed
            for embed_col in embedding_columns:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {embed_col} TEXT")
                    print(f"Created column: {embed_col}")
                except sqlite3.OperationalError:
                    print(f"Column {embed_col} already exists, will update existing values")

            conn.commit()

            for text_col, embed_col in zip(text_columns, embedding_columns):
                print(f"\nProcessing column: {text_col} -> {embed_col}")

                cursor.execute(f"SELECT id, {text_col} FROM {table_name}")
                rows = cursor.fetchall()

                # split valid vs empty texts
                valid_rows = []
                valid_texts = []
                null_row_ids = []

                for row_id, text in rows:
                    if text and text.strip():
                        valid_rows.append((row_id, text))
                        valid_texts.append(text)
                    else:
                        null_row_ids.append(row_id)

                print(f"  Valid texts: {len(valid_texts)}")
                print(f"  NULL/empty texts: {len(null_row_ids)}")

                if valid_texts:
                    print(f"  Generating embeddings...")
                    embeddings = self.embed_batch(valid_texts, normalize=normalize, batch_size=batch_size)

                    print(f"  Storing embeddings in database...")
                    for i, (row_id, _) in enumerate(valid_rows):
                        embedding_list = embeddings[i].tolist()
                        embedding_json = json.dumps(embedding_list)
                        cursor.execute(
                            f"UPDATE {table_name} SET {embed_col} = ? WHERE id = ?",
                            (embedding_json, row_id)
                        )

                    conn.commit()
                    print(f"  Successfully stored {len(valid_rows)} embeddings")

                # null out empty rows
                if null_row_ids:
                    for row_id in null_row_ids:
                        cursor.execute(
                            f"UPDATE {table_name} SET {embed_col} = NULL WHERE id = ?",
                            (row_id,)
                        )
                    conn.commit()
                    print(f"  Set NULL for {len(null_row_ids)} rows with empty text")

                stats["columns_processed"][text_col] = {
                    "embedded": len(valid_texts),
                    "skipped": len(null_row_ids)
                }

            print("\nEmbedding process completed successfully!")
            return stats

        except Exception as e:
            print(f"\nError during embedding process: {e}")
            conn.rollback()
            raise

        finally:
            conn.close()


if __name__ == "__main__":
    embedder = LegalBERTEmbedder()

    # test single text
    text = "The parties agree that jurisdiction and venue for any dispute shall be in the federal courts."
    embedding = embedder.embed(text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    # test batch
    texts = [
        "The defendant violated the terms of the contract.",
        "The court hereby grants the plaintiff's motion for summary judgment.",
        "This agreement shall be governed by the laws of the State of California."
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")

    # check cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"\nCosine similarity between first two texts: {similarity:.4f}")

    # db embedding
    print("\n" + "="*60)
    print("DATABASE EMBEDDING EXAMPLE")
    print("="*60)

    db_path = "../scotus_cases.db"
    stats = embedder.embed_database(
        db_path=db_path,
        text_columns=["case_facts", "decision"],
        embedding_columns=["case_facts_embed", "decision_embed"],
        table_name="cases",
        batch_size=32,
        normalize=True
    )

    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)
    print(f"Total rows processed: {stats['total_rows']}")
    for col, col_stats in stats['columns_processed'].items():
        print(f"\n{col}:")
        print(f"  - Embedded: {col_stats['embedded']}")
        print(f"  - Skipped (NULL/empty): {col_stats['skipped']}")
