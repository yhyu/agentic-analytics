import sys
from os.path import dirname, realpath

PROJECT_ROOT = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

from app.core.setting import settings
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("db_searcher", port=8001)

@mcp.tool()
async def get_database(hint: str = None, topN: int = 1) -> list[tuple[str, str]]:
    """Search for number of 'topN' databases and tables schemas related to 'hint'."""
    print('hint:', hint)
    # TODO: leverage semantic seach + BM25 + reranker
    return [(
        settings.DATABASE,
        settings.DB_SCHEMA
    )]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")