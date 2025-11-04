import re
import requests
from bs4 import BeautifulSoup
from typing import List, Set


def extract_case_mentions(url: str) -> List[str]:
    """
    Fetches the text from a legal document URL and extracts a unique, sorted
    list of all potential case names mentioned (e.g., 'Marbury v. Madison').
    """
    print(f"Fetching document from: {url}")

    # Set a User-Agent to mimic a web browser, which helps prevent blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # --- Step 1: Isolate the main text ---
    main_content = soup.find("article") or soup.find("main") or soup.find("body")
    document_text = main_content.get_text() if main_content else soup.get_text()

    # --- Step 2: EXTRACT POTENTIAL CASE NAMES USING GENERIC REGEX WITH LOOKAHEAD ---

    # Regex components:
    PLAINTIFF_PATTERN = r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)"
    DEFENDANT_PATTERN = r"([A-Z][a-z]+(?: [a-z]+)?(?: [A-Z][a-z]+)*)"
    # Requires the match to be followed by punctuation, a digit (citation), or a newline.
    LOOKAHEAD = r"(?=[.,;:\?!)]|\s\d|\n)"

    # CORRECTED CONCATENATION: Use '+' to join the string variables and literals.
    case_name_pattern = re.compile(
        PLAINTIFF_PATTERN + r"\s+v\.?\s+" + DEFENDANT_PATTERN + LOOKAHEAD, re.MULTILINE
    )

    all_name_parts = case_name_pattern.findall(document_text)

    # --- Step 3: Reconstruct, Clean, and Deduplicate ---
    mentioned_cases: Set[str] = set()
    CURRENT_CASE_NAME = "GIDEON V. WAINWRIGHT"
    EXCLUSION_PREFIXES = re.compile(r"^(In|The|A)\s+", re.IGNORECASE)

    for plaintiff, defendant in all_name_parts:
        # Reconstruct the full name
        full_case_name = f"{plaintiff.strip()} v. {defendant.strip()}"

        # 1. Cleanup and remove false positive punctuation
        cleaned_name = re.sub(r"[,)]$", "", full_case_name).strip()

        # 2. Filter common legal false positives (Id., See, E.g.)
        if re.search(r"\b(Id|See|E\.g)\b", cleaned_name, re.IGNORECASE):
            continue

        # 3. Filter prefixes like 'In' that create duplicates
        cleaned_name = EXCLUSION_PREFIXES.sub("", cleaned_name)

        # 4. Filter the current case being analyzed
        if cleaned_name.upper() != CURRENT_CASE_NAME:
            mentioned_cases.add(cleaned_name)

    # Convert the set of unique case names back to a sorted list
    return sorted(list(mentioned_cases))


# --- Example Usage ---

# Example URL: The full text of the Supreme Court opinion for 'Gideon v. Wainwright'
GIDEON_V_WAINWRIGHT_OPINION_URL = "https://supreme.justia.com/cases/federal/us/372/335/"

if __name__ == "__main__":
    citations_list = extract_case_mentions(GIDEON_V_WAINWRIGHT_OPINION_URL)

    print("\n" + "=" * 70)
    print("  List of Potential Case Names Mentioned (e.g., 'Roe v. Wade')")
    print("=" * 70)

    if citations_list:
        for citation in citations_list:
            print(f"📘 {citation}")
        print("\n" + "-" * 70)
        print(f"Total Unique Case Names Mentioned: {len(citations_list)}")
        print("-" * 70)
    else:
        print("No case names following the 'Name v. Name' pattern were found.")
