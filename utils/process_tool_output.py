import json

from typing import List

def process_tool_output(tool_name: str, output: str) -> str:
    """
    Process the output from a tool to ensure it fits within token limits.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the tool.

    Returns:
        str: The processed output.
    """
    tools_available = {
        "list_pull_requests": list_pull_requests,
        "list_releases": list_releases,
    }
    if tool_name in tools_available:
        return tools_available[tool_name](output)
    else:
        return f"Tool '{tool_name}' not recognized. Unprocessed output:\n{output}"


def list_pull_requests(output: str) -> List:
    """
    Specific processing for the list_pull_requests tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the list_pull_requests tool.

    Returns:
        str: The processed output.
    """
    content = getattr(output, "content", [])
    try:
        if isinstance(content, List) and content:
            text = getattr(content[0], "text", "")
        else:
            text = ""
        try:
            list = json.loads(text)
        except json.JSONDecodeError:
            return "Error: Could not list pull requests."
    except Exception as e:
        print(f"Error processing output: {e}")

    result_lines = []

    for pr in list:
        title = pr.get("title", "No title")
        url = pr.get("url", "No URL")
        state = pr.get("state", "No state")
        number = pr.get("number", "No number")

        result_lines.append(f"PR #{number} - {title} ({state}): {url}")

    return "\n".join(result_lines)

def list_releases(output: str) -> List:
    """
    Specific processing for the list_releases tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the list_releases tool.

    Returns:
        str: The processed output.
    """
    content = getattr(output, "content", [])
    try:
        if isinstance(content, List) and content:
            text = getattr(content[0], "text", "")
        else:
            text = ""
        try:
            list = json.loads(text)
        except json.JSONDecodeError:
            return "Error: Could not list releases."
    except Exception as e:
        print(f"Error processing output: {e}")

    result_lines = []

    for release in list:
        name = release.get("name", "No name")
        tag_name = release.get("tag_name", "No tag name")
        url = release.get("url", "No URL")
        published_at = release.get("published_at", "No publish date")
        body = release.get("body", "")

        result_lines.append(f"Release {name} ({tag_name}) published at {published_at}: {url}\n {body}")

    return "\n".join(result_lines)
