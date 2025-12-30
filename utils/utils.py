import re
import logging 

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
    Extracts items silently. 
    Returns: (validated_list, error_count, list_of_invalid_strings)
    """
    matched_items = []
    invalid_items = [] 
    error_count = 0

    if not content or not isinstance(content, str):
        return [], 0, []

    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    lines = [line.strip() for line in content.strip().split('\n')]

    for line in lines:
        # Matches "1. Item" or "1) Item" or just "Item" if structured poorly
        # The regex ^\s*(?:\d+[:.)]\s*)?(.*) looks for optional number followed by text
        match = re.match(r'^\s*(?:\d+[:.)]\s*)?(.*)', line)
        
        if match:
            raw_item = match.group(1).strip()
            # Skip empty lines or lines that were just numbers
            if not raw_item: 
                continue

            # Fuzzy Match
            matched_item = find_best_match(raw_item, original_items_set, score_cutoff)

            if matched_item:
                matched_items.append(matched_item)
            else:
                error_count += 1
                invalid_items.append(raw_item)
        else:
            # Fallback if regex fails completely
            error_count += 1
            invalid_items.append(line)

        if len(matched_items) == k:
            break
            
    # Return 3 values: The result, the stats, and the raw bad data
    return matched_items[:k], error_count, invalid_items


def get_correct_file_name(file_name:str)->str:
    return file_name.replace(" ", "_").replace(":", "_").replace("/", "_")
    
