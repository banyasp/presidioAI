"""
Example script demonstrating how to use case_mention_extractor
and docket_lookup together.

This script shows the complete workflow:
1. Extract case mentions from legal text
2. Look up their docket numbers from the database
"""

from utils.case_mention_extractor import extract_case_mentions_from_text
from utils.docket_lookup import get_docket_numbers


def main():
    # Example legal text
    legal_text = """
    This case follows the precedent established in Kennedy v. Braidwood Management.
    The Court's reasoning in Hewitt v. United States is also instructive.
    Similar issues were addressed in Riley v. Bondi and Mahmoud v. Taylor.
    The regulatory framework discussed in FCC v. Consumers Research applies here.
    """

    print("=" * 70)
    print("Case Mention Extraction and Docket Number Lookup Demo")
    print("=" * 70)
    print()

    # Step 1: Extract case mentions
    print("Step 1: Extracting case mentions from legal text...")
    print()
    case_names = extract_case_mentions_from_text(legal_text)

    if not case_names:
        print("No case mentions found in the text.")
        return

    print(f"Found {len(case_names)} case mention(s):")
    for i, name in enumerate(case_names, 1):
        print(f"  {i}. {name}")
    print()

    # Step 2: Look up docket numbers
    print("Step 2: Looking up docket numbers in the database...")
    print()
    docket_numbers = get_docket_numbers(case_names)

    # Step 3: Display results
    print("Results:")
    print("-" * 70)

    found_cases = []
    missing_cases = []

    for case_name, docket_number in docket_numbers.items():
        if docket_number:
            found_cases.append((case_name, docket_number))
            print(f"  ✓ {case_name}")
            print(f"    Docket Number: {docket_number}")
        else:
            missing_cases.append(case_name)
            print(f"  ✗ {case_name}")
            print(f"    Docket Number: NOT FOUND IN DATABASE")
        print()

    # Summary
    print("-" * 70)
    print(f"Summary: Found {len(found_cases)} of {len(case_names)} docket numbers")

    if missing_cases:
        print(f"\nCases not found in database ({len(missing_cases)}):")
        for case in missing_cases:
            print(f"  - {case}")

    print("=" * 70)


if __name__ == "__main__":
    main()
