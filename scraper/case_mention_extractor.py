import re
from typing import List, Set


def extract_case_mentions_from_text(document_text: str, current_case_name: str = None) -> List[str]:
    """
    Takes the raw text of a legal document and extracts a unique, sorted
    list of all potential case names mentioned (e.g., 'Marbury v. Madison').

    Args:
        document_text: The full text of the legal document as a string.
        current_case_name: Optional name of the current case to filter out self-citations.

    Returns:
        A sorted list of unique potential case names.
    """
    print("Starting case mention extraction from document text...")

    # --- Step 1: EXTRACT POTENTIAL CASE NAMES USING GENERIC REGEX WITH LOOKAHEAD ---

    # Regex components:
    # Captures a name/phrase starting with a capital letter, followed by 'v.'
    PLAINTIFF_PATTERN = r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)"
    # Captures a name/phrase for the defendant
    # Only allows specific connector words (of, the, and, for, in) that are legitimately part of case names
    DEFENDANT_PATTERN = r"([A-Z][a-z]+(?:(?: (?:of|the|and|for|in))? [A-Z][a-z]+)*)"
    # Requires the match to be followed by a closing character (punctuation, digit/citation, or a newline).
    LOOKAHEAD = r"(?=[.,;:\?!)]|\s\d|\n)"

    case_name_pattern = re.compile(
        # Matches 'Plaintiff v. Defendant' (with optional dot after v)
        PLAINTIFF_PATTERN + r"\s+v\.?\s+" + DEFENDANT_PATTERN + LOOKAHEAD,
        re.MULTILINE,
    )

    all_name_parts = case_name_pattern.findall(document_text)

    # --- Step 2: Reconstruct, Clean, and Deduplicate ---
    mentioned_cases: Set[str] = set()
    EXCLUSION_PREFIXES = re.compile(r"^(In|The|A)\s+", re.IGNORECASE)

    for plaintiff, defendant in all_name_parts:
        # Reconstruct the full name
        full_case_name = f"{plaintiff.strip()} v. {defendant.strip()}"

        # 1. Cleanup and remove false positive punctuation
        cleaned_name = re.sub(r"[,)]$", "", full_case_name).strip()

        # 2. Filter common legal false positives (Id., See, E.g.)
        if re.search(r"\b(Id|See|E\.g)\b", cleaned_name, re.IGNORECASE):
            continue

        # 3. Filter common legal prefixes like 'In' that create duplicates
        cleaned_name = EXCLUSION_PREFIXES.sub("", cleaned_name)

        # 4. Filter the current case being analyzed (if defined)
        if current_case_name and cleaned_name.upper() == current_case_name.upper():
            continue
        
        mentioned_cases.add(cleaned_name)

    # Convert the set of unique case names back to a sorted list
    return sorted(list(mentioned_cases))


# --- Example Usage ---

if __name__ == "__main__":
    # Simulate fetching and cleaning the text from a legal document
    EXAMPLE_DOCUMENT_TEXT = """
    This case, Gideon v. Wainwright, involves a fundamental question of due process.
    The prior case of Betts v. Brady established a complex rule regarding the appointment
    of counsel. This contrasts sharply with Powell v. Alabama, which mandated counsel
    in capital cases. The court in Mapp v. Ohio affirmed the exclusionary rule's application
    to the states. See also the arguments in Escobedo v. Illinois.
    The issue presented is reminiscent of earlier debates. We must overrule Betts v. Brady.
    A ruling in favor of Gideon v. Wainwright would clarify prior decisions. (Id. at 350).

    Treating due process as "a concept less rigid and more fluid than those envisaged in 
    other specific and particular provisions of the Bill of Rights," the Court held that refusal
    to appoint counsel under the particular facts and circumstances in the Betts case was not so 
    "offensive to the common and fundamental ideas of fairness" as to amount to a denial of due process. 
    Since the facts and circumstances of the two cases are so nearly indistinguishable, 
    we think the Betts v. Brady holding, if left standing, would require us to reject Gideon's claim 
    that the Constitution guarantees him the assistance of counsel. Upon full reconsideration, we conclude 
    that Betts v. Brady should be overruled.

    E.g., Gitlow v. New York, 268 U. S. 652, 268 U. S. 666 (1925) (speech and press);
    Lovell v. City of Griffin, 303 U. S. 444, 303 U. S. 450 (1938) (speech and press); 
    Staub v. City of Baxley, 355 U. S. 313, 355 U. S. 321 (1958) (speech); Grosjean v. American Press Co., 
    297 U. S. 233, 297 U. S. 244 (1936) (press); Cantwell v. Connecticut, 310 U. S. 296, 310 U. S. 303 (1940) 
    (religion); De Jonge v. Oregon, 299 U. S. 353, 299 U. S. 364 (1937) (assembly); Shelton v. Tucker, 364 U. S. 
    479, 364 U. S. 486, 488 (1960) (association); Louisiana ex rel. Gremillion v. NAACP, 366 U. S. 293, 366 U. S. 
    296 (1961) (association); Edwards v. South Carolina, 372 U. S. 229 (1963) (speech, assembly, petition for 
    redress of grievances).
    """

    citations_list = extract_case_mentions_from_text(EXAMPLE_DOCUMENT_TEXT)

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
