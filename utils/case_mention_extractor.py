import re
import requests
from bs4 import BeautifulSoup
from typing import List, Set


def extract_case_mentions(url: str) -> List[str]:
    """
    Fetches the text from a legal document URL and extracts a unique, sorted
    list of all potential case names mentioned (e.g., 'Marbury v. Madison').

    It searches for capitalized words separated by 'v.' or 'v' within the text.
    """
    print(f"Fetching document from: {url}")

    # Set a User-Agent to mimic a web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # --- Step 1: Isolate the main text ---
    # Find the most likely tag for the main opinion text
    main_content = soup.find("article") or soup.find("main") or soup.find("body")

    if not main_content:
        document_text = soup.get_text()
    else:
        document_text = main_content.get_text()

    # --- Step 2: Extract potential case names using Regular Expression ---

    # Regex to find potential case names:
    # It looks for 1 or more capitalized words, followed by ' v. ' or ' v ',
    # followed by 1 or more capitalized words.
    # The pattern is very broad to capture many potential case names.
    case_name_pattern = re.compile(
        r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?: [a-z]+)?(?: [A-Z][a-z]+)*)",
        re.MULTILINE,
    )

    # Find all matches (returns tuples of (Plaintiff_Name, Defendant_Name))
    all_name_parts = case_name_pattern.findall(document_text)

    # --- Step 3: Reconstruct and deduplicate (MODIFIED CLEANUP) ---
    mentioned_cases: Set[str] = set()
    CURRENT_CASE_NAME = "GIDEON V. WAINWRIGHT"
    EXCLUSION_PREFIXES = re.compile(r"^(In|The|A)\s+", re.IGNORECASE)

    # NEW: Regex to find and remove common single-word artifacts at the end of a match
    # This removes words like: be, to, holding, rule, should, etc.
    TRAILING_ARTIFACTS = re.compile(
        r"\s+(be|to|made|rule|holding|should|departed|was|is|an|a)\s*$", re.IGNORECASE
    )

    for plaintiff, defendant in all_name_parts:
        full_case_name = f"{plaintiff.strip()} v. {defendant.strip()}"

        # 1. Cleanup and remove false positive punctuation
        cleaned_name = re.sub(r"[,)]$", "", full_case_name).strip()

        # 2. Filter common legal false positives (Id., See, E.g.)
        if re.search(r"\b(Id|See|E\.g)\b", cleaned_name, re.IGNORECASE):
            continue

        # 3. Filter prefixes like 'In' that create duplicates
        cleaned_name = EXCLUSION_PREFIXES.sub("", cleaned_name)

        # 4. NEW: Strip the trailing single-word artifacts
        cleaned_name = TRAILING_ARTIFACTS.sub("", cleaned_name).strip()

        # 5. Filter the current case being analyzed
        if cleaned_name.upper() != CURRENT_CASE_NAME:
            mentioned_cases.add(cleaned_name)

    return sorted(list(mentioned_cases))


# --- Example Usage ---

# Example URL: The full text of the Supreme Court opinion for 'Gideon v. Wainwright',
# a case with many mentions, hosted on Justia.
GIDEON_V_WAINWRIGHT_OPINION_URL = "https://supreme.justia.com/cases/federal/us/372/335/"

if __name__ == "__main__":
    case_mentions_list = extract_case_mentions(GIDEON_V_WAINWRIGHT_OPINION_URL)

    print("\n" + "=" * 70)
    print("  List of Potential Case Names Mentioned (e.g., 'Roe v. Wade')")
    print("=" * 70)

    if case_mentions_list:
        for name in case_mentions_list:
            print(f"📘 {name}")
        print("\n" + "-" * 70)
        print(f"Total Unique Case Names Mentioned: {len(case_mentions_list)}")
        print("-" * 70)
    else:
        print("No case names following the 'Name v. Name' pattern were found.")
