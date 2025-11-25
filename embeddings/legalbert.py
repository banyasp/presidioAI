"""LegalBERT embedding module for generating vector representations of legal text."""

import torch
import numpy as np
import sqlite3
import json
from transformers import AutoModel, AutoTokenizer


class LegalBERTEmbedder:
    """
    A class for generating embeddings from legal text using LegalBERT.

    The model is loaded once and reused for all subsequent embedding operations.
    Automatically detects and uses the best available device (CUDA > MPS > CPU).
    """

    def __init__(
        self,
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize the LegalBERT embedder.

        Args:
            device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = "nlpaueb/legal-bert-base-uncased"
        self.max_length = max_length

        # Auto-detect device
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

        # Lazy loading - models will be loaded on first use
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self._model is None:
            print(f"Loading model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")

    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate an embedding for a single text paragraph.

        Args:
            text: Input text to embed
            normalize: Whether to L2-normalize the embedding (recommended for cosine similarity)

        Returns:
            NumPy array of shape (768,) containing the embedding
        """
        self._load_model()

        # Tokenize input
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Extract CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        # Convert to NumPy
        embedding = embedding.cpu().numpy()

        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple text paragraphs efficiently.

        Args:
            texts: List of input texts to embed
            normalize: Whether to L2-normalize the embeddings
            batch_size: Number of texts to process at once

        Returns:
            NumPy array of shape (n, 768) containing the embeddings
        """
        self._load_model()

        all_embeddings = []

        # Process in batches (for efficiency)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs) # pass the tokenized inputs to the model
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # pulls out the CLS embedding vector

            all_embeddings.append(batch_embeddings) # add to a list

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        # Normalize if we want (will help with cosine similarity bc then it's just a dot product)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    def embed_database(
        self,
        db_path: str,
        text_columns: list[str],
        embedding_columns: list[str],
        table_name: str = "cases",
        batch_size: int = 32,
        normalize: bool = True
    ) -> dict:
        """
        Generate embeddings for text columns in a SQLite database and store them.

        Args:
            db_path: Path to the SQLite database
            text_columns: List of column names containing text to embed
            embedding_columns: List of column names to store embeddings (must match text_columns length)
            table_name: Name of the table to process
            batch_size: Batch size for embedding generation
            normalize: Whether to L2-normalize the embeddings

        Returns:
            Dictionary with statistics about the embedding process
        """
        if len(text_columns) != len(embedding_columns):
            raise ValueError("text_columns and embedding_columns must have the same length")

        print(f"\nConnecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        stats = {
            "total_rows": 0,
            "columns_processed": {}
        }

        try:
            # Get total number of rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats["total_rows"] = cursor.fetchone()[0]
            print(f"Total rows in {table_name}: {stats['total_rows']}")

            # Add embedding columns if they don't exist
            for embed_col in embedding_columns:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {embed_col} TEXT")
                    print(f"Created column: {embed_col}")
                except sqlite3.OperationalError:
                    print(f"Column {embed_col} already exists, will update existing values")

            conn.commit()

            # Process each text column
            for text_col, embed_col in zip(text_columns, embedding_columns):
                print(f"\nProcessing column: {text_col} -> {embed_col}")

                # Fetch all rows with their IDs and text
                cursor.execute(f"SELECT id, {text_col} FROM {table_name}")
                rows = cursor.fetchall()

                # Separate valid texts from NULL/empty ones
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

                # Generate embeddings for valid texts
                if valid_texts:
                    print(f"  Generating embeddings...")
                    embeddings = self.embed_batch(
                        valid_texts,
                        normalize=normalize,
                        batch_size=batch_size
                    )

                    # Update database with embeddings
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

                # Set NULL for rows with NULL/empty text
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


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    embedder = LegalBERTEmbedder()

    # Single text embedding
    text = "The parties agree that jurisdiction and venue for any dispute shall be in the federal courts."
    embedding = embedder.embed(text)
    print(f"\nSingle embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    # Batch embedding
    texts = [
        "The defendant violated the terms of the contract.",
        "The court hereby grants the plaintiff's motion for summary judgment.",
        "This agreement shall be governed by the laws of the State of California."
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")

    # Demonstrate cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f"\nCosine similarity between first two texts: {similarity:.4f}")

    # Database embedding example
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
