import re
import logging # Import the logging library

# --- Configure Logging ---
# Set up logging to write matches to a file
# Use 'a' mode to append, create file if it doesn't exist
# Use a different filename than your error log
logging.basicConfig(filename='fuzzy_matches.log',
                    level=logging.INFO, # Log INFO level messages and above
                    format='%(asctime)s - %(message)s', # Include timestamp
                    datefmt='%Y-%m-%d %H:%M:%S',
                    encoding='utf-8', # Use utf-8 encoding
                    filemode='a') # Append mode
      
# utils.py (or wherever your functions reside)

import re
from thefuzz import process, fuzz
import numpy as np

def find_best_match(extracted_item, original_items_set, score_cutoff=89):
    """
    Uses fuzzy matching to find the best match for an extracted item
    within the original set of items. Includes preprocessing.

    Args:
        extracted_item (str): The item string extracted from the LLM response.
        original_items_set (set): A set containing the valid items from the original dataset.
        score_cutoff (int): The minimum similarity score (0-100) required to consider it a match.

    Returns:
        str | None: The best matching string from original_items_set if found above cutoff,
                    otherwise None.
    """
    if not original_items_set or not extracted_item:
        return None # Cannot match against an empty set or empty item

    processed_item = extracted_item.lower().strip()
    # Remove common additions like (MIT), (UCLA), etc. - adjust regex as needed
    processed_item = re.sub(r'\s*\([^)]*\)$', '', processed_item).strip()
    processed_item = re.sub(r'[.,;:!*]$', '', processed_item).strip()
    processed_item = processed_item.strip("'\"`")
    # Attempt to remove descriptions like ", Brazil – *Excellent..." or ", Brazil - *Excellent..."
    # Handles both en-dash and regular dash after country/region name
    processed_item = re.split(r'\s*,\s*[\w\s]+[\u2013-]\s*\*.*$', processed_item, 1)[0].strip()
    # Remove potential leading list markers if regex didn't catch them perfectly (e.g., "1. ")
    processed_item = re.sub(r'^\s*\d+[:.]\s*', '', processed_item).strip()


    if not processed_item:
        print(f"Warning: Processed item is empty after regex cleanup: '{extracted_item}'")
        return None

    # Find the best match in the original set
    result = process.extractOne(
        processed_item,
        original_items_set,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff
        )
    if result:
        best_match, score = result
        # logging.info(f"MATCH FOUND: Extracted='{extracted_item}' | Processed='{processed_item}' | Matched To='{best_match}' | Score={score}")
        return best_match
    else:
        return None

def extract_list_from_response(content: str, original_items_set: set, k: int = 10, score_cutoff: int = 88):
    """
    Extracts items from a simple numbered list format (1. Item\n2. Item...).
    Performs fuzzy matching against original_items_set for validation.

    Args:
        content (str): The raw text response from the LLM.
        original_items_set (set): A set containing the valid items from the original dataset.
        k (int): The maximum number of items to extract.
        score_cutoff (int): Minimum fuzzy matching score required (0-100).

    Returns:
        tuple[list[str], int]: A tuple containing:
            - list: The list of *validated* item names (matching original set).
            - int: The count of extracted lines that *failed* fuzzy matching.
    """
    matched_items = []
    error_count = 0
    invalid_items = []
    processed_lines = 0

    if not content or not isinstance(content, str):
        print("Warning: Received invalid content for extraction.")
        return [], 0

    lines = [line.strip() for line in content.strip().split('\n')]

    for line in lines:
        # Basic regex for "number. item_name" or "number: item_name"
        # Makes the number part optional to catch lists that might lose numbering
        match = re.match(r'^\s*(?:\d+[:.]\s*)?(.*)', line)
        processed_lines += 1
        if match:
            raw_item = match.group(1).strip()
            if not raw_item: # Skip empty lines or lines with only numbering
                continue

           
            matched_item = find_best_match(raw_item, original_items_set, score_cutoff)

            if matched_item:
                matched_items.append(matched_item)
            else:
                error_count += 1
                invalid_items.append(raw_item) # Collect invalid items for error reporting
        else:
            error_count += 1
            invalid_items.append(raw_item)

        if len(matched_items) == k:
            break
    
    if error_count > 0:
        # Append errors to a file
        with open("errors.txt", "a") as error_file:
            print("-" * 20 + "\n")
            print(f"Format Errors Found: {error_count} out of {processed_lines}\n")
            for invalid_item in invalid_items:
                error_file.write(f"'{invalid_item}'\n")
            print("-" * 20 + "\n")


    return matched_items[:k], error_count


def get_correct_file_name(file_name:str)->str:
    return file_name.replace(" ", "_").replace(":", "_").replace("/", "_")
    
