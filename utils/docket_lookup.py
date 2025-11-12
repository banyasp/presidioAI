"""
Utility functions for looking up docket numbers from case names.
"""

import sqlite3
import re
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_case_name(case_name: str) -> str:
    """
    Normalize a case name for comparison by removing punctuation,
    corporate suffixes, extra whitespace, and converting to lowercase.

    Args:
        case_name: The case name to normalize (e.g., "Trump v. CASA, Inc.")

    Returns:
        Normalized case name (e.g., "trump v casa")

    Examples:
        >>> normalize_case_name("Trump v. CASA, Inc.")
        'trump v casa'
        >>> normalize_case_name("Kennedy v. Braidwood Management, Inc.")
        'kennedy v braidwood management'
        >>> normalize_case_name("Betts v. Brady")
        'betts v brady'
    """
    # Convert to lowercase
    normalized = case_name.lower()

    # Remove periods (except in 'v.')
    normalized = normalized.replace('v.', 'v')

    # Remove common punctuation (commas, periods, quotes, etc.)
    normalized = re.sub(r'[,.\'"();\[\]{}]', '', normalized)

    # Remove common corporate/legal suffixes (Inc, Co, LLC, Corp, Ltd, LLP, etc.)
    # This helps match "Kennedy v. Braidwood Management" with "Kennedy v. Braidwood Management, Inc."
    suffixes = r'\b(inc|co|llc|corp|corporation|ltd|limited|lp|llp)\b'
    normalized = re.sub(suffixes, '', normalized, flags=re.IGNORECASE)

    # Normalize whitespace (replace multiple spaces with single space)
    normalized = re.sub(r'\s+', ' ', normalized)

    # Strip leading/trailing whitespace
    normalized = normalized.strip()

    return normalized


def get_docket_numbers(
    case_names: List[str],
    db_path: str = './scotus_cases.db'
) -> Dict[str, Optional[str]]:
    """
    Look up docket numbers for a list of case names from the database.

    Uses normalized matching to handle formatting differences between
    case names extracted from text and case names in the database.

    Args:
        case_names: List of case names (e.g., from case_mention_extractor)
        db_path: Path to the SQLite database (default: './scotus_cases.db')

    Returns:
        Dictionary mapping case names to their docket numbers.
        Returns None for cases not found in the database.

    Examples:
        >>> case_names = ["Betts v. Brady", "Powell v. Alabama"]
        >>> get_docket_numbers(case_names)
        {'Betts v. Brady': '316', 'Powell v. Alabama': '98'}

    Raises:
        FileNotFoundError: If the database file doesn't exist
        sqlite3.Error: If there's a database connection error
    """
    result = {}

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all cases from database
        cursor.execute("SELECT case_name, docket_number FROM cases")
        db_cases = cursor.fetchall()

        # Create a normalized lookup dictionary
        # Maps normalized_case_name -> (original_db_case_name, docket_number)
        normalized_db_lookup = {}
        for db_case_name, docket_number in db_cases:
            normalized_db_case = normalize_case_name(db_case_name)
            normalized_db_lookup[normalized_db_case] = (db_case_name, docket_number)

        # Look up each case name
        for case_name in case_names:
            normalized_input = normalize_case_name(case_name)

            if normalized_input in normalized_db_lookup:
                # Found a match
                _, docket_number = normalized_db_lookup[normalized_input]
                result[case_name] = docket_number
                logger.debug(f"Found docket number '{docket_number}' for case '{case_name}'")
            else:
                # No match found
                result[case_name] = None
                logger.warning(f"No docket number found for case '{case_name}'")

        conn.close()

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Database file not found: {db_path}")
        raise

    return result


if __name__ == "__main__":
    # Example usage
    import sys

    # Test with some example case names
    test_cases = [
        "Betts v. Brady",
        "Powell v. Alabama",
        "Mapp v. Ohio",
        "NonExistent v. Case"  # This one won't be found
    ]

    print("Testing docket number lookup...")
    print(f"Input cases: {test_cases}\n")

    try:
        docket_numbers = get_docket_numbers(test_cases)
        print("Results:")
        for case_name, docket_number in docket_numbers.items():
            if docket_number:
                print(f"  {case_name} → {docket_number}")
            else:
                print(f"  {case_name} → NOT FOUND")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
