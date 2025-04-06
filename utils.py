import re


def extract_list_from_response(response):
    try:
        # Extract the list from the content
        content = response["message"]["content"].split("[")[1].split("]")[0]
        matches = re.findall(r"\d+[:.]\s(.*?)(?=, \d+:|$)", content)
        if matches:
            return [item.strip() for item in matches]
        else:
            return []
    except:
        return []
