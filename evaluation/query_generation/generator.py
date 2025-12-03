"""Query generation from legal case facts."""

import re
import sqlite3
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging

from .ollama_client import call_ollama
from ..storage.database import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_factual_query(case_facts: str) -> str:
    """
    Extract a factual query from case facts using rule-based methods.

    This uses simple heuristics to extract key elements:
    - Key legal concepts and terms
    - Parties and their roles
    - Core legal issues

    Args:
        case_facts: Case facts text

    Returns:
        Extracted factual query string
    """
    if not case_facts or len(case_facts.strip()) == 0:
        return ""

    # Clean up the text
    text = case_facts.strip()

    # Skip header material (Syllabus, case names, docket info, dates)
    # Look for paragraphs that contain substantive content
    paragraphs = text.split('\n')

    # Filter out header lines
    content_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip obvious headers
        if (para.startswith('Syllabus') or
            para.startswith('No.') or
            re.match(r'^[A-Z\s,\.]+v\.[A-Z\s,\.]+$', para) or  # Case names
            re.match(r'.*Argued.*Decided.*', para) or  # Date line
            'on application' in para.lower() or
            'on petition' in para.lower() or
            len(para.split()) < 5):  # Very short lines
            continue

        content_paragraphs.append(para)

    # Join paragraphs and extract sentences
    content_text = ' '.join(content_paragraphs)

    if not content_text:
        # Fallback: use original text if filtering removed everything
        content_text = text

    # Extract sentences
    sentences = re.split(r'[.!?]+', content_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

    # Take first 2-3 sentences or up to 150 words
    query_parts = []
    word_count = 0
    max_words = 150

    for sentence in sentences[:5]:  # Look at more sentences
        words = sentence.split()
        if word_count + len(words) <= max_words:
            query_parts.append(sentence)
            word_count += len(words)
        else:
            # Take partial sentence to reach max_words
            remaining = max_words - word_count
            if remaining > 10:  # Only add if meaningful
                query_parts.append(' '.join(words[:remaining]))
            break

        # Stop if we have at least 50 words and this sentence ends a clause
        if word_count >= 50:
            break

    query = ' '.join(query_parts)

    # Ensure it ends with punctuation
    if query and query[-1] not in '.!?':
        query += '.'

    return query


def generate_legal_question(case_facts: str, model: str = "llama3.2:1b") -> Optional[str]:
    """
    Generate a natural legal question from case facts using Ollama.

    Args:
        case_facts: Case facts text
        model: Ollama model name

    Returns:
        Generated legal question or None if generation failed
    """
    if not case_facts or len(case_facts.strip()) == 0:
        return None

    # Truncate case_facts to avoid very long prompts
    max_facts_length = 1000
    facts_truncated = case_facts[:max_facts_length]
    if len(case_facts) > max_facts_length:
        facts_truncated += "..."

    prompt = f"""Convert this legal case summary into a natural question that someone might search for. Make it concise (1-2 sentences) and focus on the core legal issue.

Case summary:
{facts_truncated}

Legal question:"""

    result = call_ollama(prompt, model=model)

    if result:
        # Clean up the result
        result = result.strip()

        # Remove any "A:" or "Q:" prefixes
        result = re.sub(r'^[AQ]:\s*', '', result, flags=re.IGNORECASE)

        # Ensure it ends with a question mark if it looks like a question
        if result and '?' not in result and any(result.lower().startswith(q) for q in ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'does', 'is']):
            result += '?'

        return result

    return None


def generate_queries_for_all_cases(
    db_path: str,
    model: str = "llama3.2:1b",
    use_extractive: bool = True,
    use_generative: bool = True
) -> int:
    """
    Generate queries for all cases in the database.

    Args:
        db_path: Path to SQLite database
        model: Ollama model name for generative queries
        use_extractive: Generate extractive queries
        use_generative: Generate generative queries

    Returns:
        Number of queries generated
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get all cases
    cursor.execute("SELECT id, case_name, case_facts FROM cases")
    cases = cursor.fetchall()

    logger.info(f"Found {len(cases)} cases in database")

    queries_generated = 0

    # Check if Ollama is needed and available
    if use_generative:
        from .ollama_client import OllamaClient
        ollama_client = OllamaClient(model=model)
        if not ollama_client.is_available():
            logger.error("Ollama is not running! Generative queries will be skipped.")
            logger.info("Start Ollama with: ollama serve")
            use_generative = False

    for case in tqdm(cases, desc="Generating queries"):
        case_id = case['id']
        case_name = case['case_name']
        case_facts = case['case_facts']

        if not case_facts or len(case_facts.strip()) == 0:
            logger.warning(f"Skipping case {case_id} ({case_name}): no case_facts")
            continue

        # Generate extractive query
        if use_extractive:
            extractive_query = extract_factual_query(case_facts)
            if extractive_query:
                cursor.execute("""
                    INSERT INTO evaluation_queries (source_case_id, query_text, query_type, generation_method)
                    VALUES (?, ?, ?, ?)
                """, (case_id, extractive_query, 'extractive', 'regex'))
                queries_generated += 1

        # Generate generative query
        if use_generative:
            generative_query = generate_legal_question(case_facts, model=model)
            if generative_query:
                cursor.execute("""
                    INSERT INTO evaluation_queries (source_case_id, query_text, query_type, generation_method)
                    VALUES (?, ?, ?, ?)
                """, (case_id, generative_query, 'generative', f'ollama-{model}'))
                queries_generated += 1
            else:
                logger.warning(f"Failed to generate query for case {case_id}")

    conn.commit()
    conn.close()

    logger.info(f"✓ Generated {queries_generated} queries")
    return queries_generated


def get_queries_by_type(db_path: str) -> Dict[str, List[Dict]]:
    """
    Get all queries grouped by type.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary with 'extractive' and 'generative' query lists
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            eq.id,
            eq.source_case_id,
            eq.query_text,
            eq.query_type,
            eq.generation_method,
            c.case_name
        FROM evaluation_queries eq
        JOIN cases c ON eq.source_case_id = c.id
        ORDER BY eq.query_type, eq.id
    """)

    rows = cursor.fetchall()
    conn.close()

    result = {
        'extractive': [],
        'generative': []
    }

    for row in rows:
        query_info = {
            'id': row['id'],
            'source_case_id': row['source_case_id'],
            'source_case_name': row['case_name'],
            'query_text': row['query_text'],
            'generation_method': row['generation_method']
        }
        result[row['query_type']].append(query_info)

    return result


def print_sample_queries(db_path: str, n: int = 5) -> None:
    """
    Print sample queries for inspection.

    Args:
        db_path: Path to SQLite database
        n: Number of samples per type
    """
    queries = get_queries_by_type(db_path)

    print(f"\n{'='*80}")
    print("SAMPLE QUERIES")
    print(f"{'='*80}\n")

    for query_type in ['extractive', 'generative']:
        print(f"\n{query_type.upper()} QUERIES:")
        print(f"{'-'*80}")

        samples = queries[query_type][:n]
        for i, q in enumerate(samples, 1):
            print(f"\n{i}. Source: {q['source_case_name']}")
            print(f"   Query: {q['query_text'][:200]}")
            if len(q['query_text']) > 200:
                print(f"          ...")

    print(f"\n{'='*80}")
    print(f"Total extractive: {len(queries['extractive'])}")
    print(f"Total generative: {len(queries['generative'])}")
    print(f"TOTAL: {len(queries['extractive']) + len(queries['generative'])}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generator.py <path_to_db> [--extractive-only] [--generative-only]")
        sys.exit(1)

    db_path = sys.argv[1]

    use_extractive = '--generative-only' not in sys.argv
    use_generative = '--extractive-only' not in sys.argv

    print(f"Generating queries from {db_path}")
    print(f"Extractive: {use_extractive}, Generative: {use_generative}")

    count = generate_queries_for_all_cases(
        db_path,
        use_extractive=use_extractive,
        use_generative=use_generative
    )

    print(f"\n✓ Generated {count} queries")

    # Print samples
    print_sample_queries(db_path)
