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
        "list_pull_requests": list_pull_requests
    }
    if tool_name in tools_available:
        return tools_available[tool_name](output)
    else:
        return f"Tool '{tool_name}' not recognized."


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
        list = json.loads(text)
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
