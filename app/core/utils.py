from duckduckgo_search import DDGS


def web_search(query: str):
    """Search the web for information"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            if not results:
                return "Error: no results found."

            formatted_results = []
            for result in results:
                formatted_results.append(f"Title: {result['title']}\nBody: {result['body']}\n")

            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error performing search: {str(e)}"
