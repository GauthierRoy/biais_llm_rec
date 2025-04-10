import re


def extract_list_from_response(content) -> list:
    try:
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
            print("Error: No matches found in the response.")
            print(f"Response content: {content}")
            return []
    except:
        print("Error: Unable to extract list from response.")
        print(f"Response content: {content}")
        return []
