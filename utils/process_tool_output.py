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
        "list_issues": list_issues,
        "get_file_contents": get_file_contents,
        "get_pull_request": get_pull_request,
        "get_issue": get_issue,
        "get_release_by_tag": get_release_by_tag
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
        except Exception as e:
            return f"Error: Could not list pull requests. {e} Try again by using just the required parameters."
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
            return "Error: Could not list releases. Try again by using just the required parameters."
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


def list_issues(output: str) -> List:
    """
    Specific processing for the list_issues tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the list_issues tool.

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
            return "Error: Could not list issues. Try again modifying the parameters."
    except Exception as e:
        print(f"Error processing output: {e}")

    result_lines = []

    issues = list.get("issues", [])

    for issue in issues:
        title = issue.get("title", "No title")
        state = issue.get("state", "No state")
        number = issue.get("number", "No number")
        body = issue.get("body", "No body")

        result_lines.append(f"Issue #{number} - {title} ({state})")

    return "\n".join(result_lines)

def get_file_contents(output: str) -> str:
    """
    Specific processing for the get_file_contents tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the get_file_contents tool.

    Returns:
        str: The processed output.
    """
    content = getattr(output, "content", [])
    try:
        if isinstance(content, List) and content:
            resource = getattr(content[1], "resource", "")
            text = getattr(resource, "text", "")
        else:
            text = ""
    except Exception as e:
        return f"Error processing output: {e}. Try again by using just the required parameters or change ref parameter from 'main' to 'master'."

    return text

def get_pull_request(output: str) -> str:
    """
    Specific processing for the get_pull_request tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the get_pull_request tool.

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
            pr = json.loads(text)
        except json.JSONDecodeError:
            return "Error: Could not get pull request. Try again by using just the required parameters."
    except Exception as e:
        print(f"Error processing output: {e}")

    title = pr.get("title", "No title")
    url = pr.get("url", "No URL")
    state = pr.get("state", "No state")
    number = pr.get("number", "No number")
    body = pr.get("body", "No body")

    if len(body) > 5000:
        body = body[:5000] + "..."

    return f"PR #{number} - {title} ({state}): {url}\n {body}"


def get_issue(output: str) -> str:
    """
    Specific processing for the get_issue tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the get_issue tool.

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
            issue = json.loads(text)
        except json.JSONDecodeError:
            return "Error: Could not get issue. Try again by using just the required parameters."
    except Exception as e:
        print(f"Error processing output: {e}")

    title = issue.get("title", "No title")
    url = issue.get("url", "No URL")
    state = issue.get("state", "No state")
    number = issue.get("number", "No number")
    body = issue.get("body", "No body")

    if len(body) > 5000:
        body = body[:5000] + "..."

    return f"Issue #{number} - {title} ({state}): {url}\n {body}"

def get_release_by_tag(output: str) -> str:
    """
    Specific processing for the get_release_by_tag tool output.
    This function can be customized based on specific needs.

    Args:
        output (str): The raw output from the get_release_by_tag tool.

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
            release = json.loads(text)
        except json.JSONDecodeError:
            return "Error: Could not get release by tag. Try again by using just the required parameters."
    except Exception as e:
        print(f"Error processing output: {e}")

    name = release.get("name", "No name")
    tag_name = release.get("tag_name", "No tag name")
    url = release.get("url", "No URL")
    published_at = release.get("published_at", "No publish date")
    body = release.get("body", "")

    if len(body) > 5000:
        body = body[:5000] + "..."

    return f"Release {name} ({tag_name}) published at {published_at}: {url}\n {body}"