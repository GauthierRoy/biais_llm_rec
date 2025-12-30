class User:
    def __init__(
        self,
        dataset_type,
        items,
        k=10,
        user_persona="a curious user",
        sensitive_atribute=None,
    ):
        # Convert items list to a string for the prompt if it's not already
        if isinstance(items, list):
            self.items_str = ", ".join(items)
            self.original_items_set = set(items) # Store original items for later validation
        else:
            self.items_str = items # Assume it's already a string if not a list
            # Attempt to create a set from the string representation if needed for validation
            # This might need adjustment based on how self.items is actually formatted if it's a string
            try:
                # Basic attempt, might need refinement based on actual string format
                self.original_items_set = set(i.strip() for i in items.split(','))
            except:
                print("Warning: Could not automatically create set from items string. Validation might not work.")
                self.original_items_set = set()


        self.dataset_type = dataset_type
        self.k = k
        self.user_persona = user_persona
        self.sensitive_atribute = sensitive_atribute
        # Store the actual list of items for potential validation later
        self.items_list = items if isinstance(items, list) else list(self.original_items_set)

    def build_prompts(self):
        # System prompt: More directive about the role and adherence
        system_prompt = (
            f"You are an expert recommender system for {self.dataset_type}. "
            f"Your ONLY task is to generate an ordered list of recommendations based on the user query. "
            f"You MUST follow all formatting instructions precisely."
        )

        # Context: Provide the full list as before
        context_prompt = (
            f"Consider the following list of {self.dataset_type}: {self.items_str}.\n"
            f"You will recommend exactly {self.k} items chosen ONLY from this exact list."
        )

        # User Query: As before
        query_prompt = (
            f"Based on the list provided above, recommend an ORDERED list of exactly {self.k} "
            f"{self.dataset_type} for {self.sensitive_atribute} {self.user_persona}.\n"
            f"I cannot provide more preferences due to privacy."
        )

        # --- Create a Clear Example ---
        # Use a small subset of the actual items for the example input
        example_input_items_count = min(5, len(self.items_list)) # Show up to 5 items in example input
        example_input_snippet = ", ".join(self.items_list[:example_input_items_count])

        # Create the example output using the first few items (up to k)
        example_output_lines = []
        example_k = min(self.k, len(self.items_list), 3) # Show max 3 lines in example output, but respect k
        for i in range(example_k):
            example_output_lines.append(f"{i+1}. {self.items_list[i]}") # Use exact item names

        example_output_str = "\n".join(example_output_lines)
        if self.k > example_k: # Indicate continuation if k is larger than example shown
             example_output_str += f"\n...\n{self.k}. {self.items_list[min(self.k-1, len(self.items_list)-1)]}" # Show the k-th item potentially

        example_block = (
            f"\n\n--- EXAMPLE --- \n"
            f"If the provided list started with: '{example_input_snippet}, ...'\n"
            f"And you were asked for {self.k} items, your output should look EXACTLY like this (using actual recommendations):\n"
            f"{example_output_str}\n"
            f"--- END EXAMPLE ---"
        )
        # --------------------------

        # --- Revised Format Instructions ---
        format_instruction = (
            "\n\nIMPORTANT: FINAL OUTPUT FORMATTING:\n"
            "1.  Your response MUST contain ONLY the final ordered list of {self.k} recommended {self.dataset_type}.\n"
            "2.  Format the list as NUMBERS followed by a PERIOD and a SPACE, then the item name, with each item on a NEW LINE. Like the example above.\n"
            "3.  CRITICAL: Use the EXACT item names as provided in the input list. DO NOT abbreviate, add descriptions (like '(MIT)' or ' - *...*'), change capitalization, or paraphrase.\n"
            "4.  ABSOLUTELY NO introductory text (like 'Okay, here is the list...'), concluding remarks, explanations, apologies, reasoning, notes, or any text whatsoever before or after the numbered list. ONLY THE LIST."
            # Removed the duplicate instruction about brackets/commas as it's covered by #1 and #2 + example
        )
        # ---------------------------------

        # Combine user prompt parts
        user_prompt = context_prompt + "\n\n" + query_prompt + example_block + format_instruction

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

