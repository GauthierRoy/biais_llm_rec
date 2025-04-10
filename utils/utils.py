import re


def extract_list_from_response(response):
    try:
        # Extract the list from the content
        content = response["message"]["content"]
        # remove some unwanted characters
        content = content.replace('"', "")
        content = content.replace("'", "")
        content = content.replace("**", "")
        # remove inside parenthesis
        content = re.sub(r"\s*\(.*?\)", "", content)
        if "[" in content:
            content = "".join(content.split("[")[1:])
        if "]" in content:
            content = "".join(content.split("]")[:-1])

        content = content.replace("\n\n ", ", ")
        content = content.replace("\n\n", ", ")
        content = content.replace("\n ", ", ")
        content = content.replace("\n", ", ")
        matches = re.findall(r"\d+[:.]\s(.*?)(?=, \d+[:.]|$)", content)

        if matches:
            return [item.strip() for item in matches]
        else:
            return []
    except:
        return []
