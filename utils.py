import re


def extract_list_from_response(response):
    # Extract the list from the content
    content = response["message"]["content"]
    match = re.search(r"\[(.*?)\]", content)
    if match:
        extracted_list = match.group(1).split(", ")
        return extracted_list

    match = re.findall(r"\d+\.\s(.+)", content)
    if match:
        return [item.strip() for item in match]
    else:
        return []
