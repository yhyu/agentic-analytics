import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from joblib import Memory
from duckduckgo_search import DDGS
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.setting import settings, logger


cache_loc = os.environ.get(
    'DEEP_ANALYTICS_CACHE_LOC',
    default=os.path.join(Path.home(), 'cache_DeepAnalytics')
)
MemoryCache = Memory(location=cache_loc, verbose=0)


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


class DBInfo:
    def __init__(self):
        self.mcp_get_database = None
        self.get_database = MemoryCache.cache(self._get_database)

    async def _init_mcp(self):
        if not settings.DB_SEARCH_MCP_URL or self.mcp_get_database:
            return

        if settings.DB_SEARCH_MCP_URL:
            client = MultiServerMCPClient(
                {
                    settings.DB_SEARCH_MCP_NAME: {
                        "transport": settings.DB_SEARCH_MCP_TRANSPORT,
                        "url": settings.DB_SEARCH_MCP_URL,
                        "headers": settings.DB_SEARCH_MCP_HEADER,
                    }
                }
            )
            try:
                mcp_tools = await client.get_tools()
                mcp_db_search_tool = [t for t in mcp_tools if t.name == settings.DB_SEARCH_MCP_TOOL]
                if mcp_db_search_tool:
                    self.mcp_get_database = (lambda **kw: mcp_db_search_tool[0].ainvoke(kw))
                    logger.info('mcp tool is initialized.')
            except Exception:
                pass

    async def _get_database(self, hint: str = None, topN: int = 1) -> list[tuple[str, str]]:
        # Use MCP
        await self._init_mcp()
        if self.mcp_get_database:
            return await self.mcp_get_database(hint=hint, topN=topN)

        # TODO: leverage semantic seach + BM25 + reranker
        return [(
            settings.DATABASE,
            settings.DB_SCHEMA
        )]
