from threading import RLock
from typing import Any, Dict

from duckduckgo_search import DDGS


class Singleton(type):
    _instances: Dict = {}
    _lock: RLock = RLock()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with self._lock:
            if self not in self._instances:
                instance = super().__call__(*args, **kwds)
                self._instances[self] = instance
        return self._instances[self]


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
