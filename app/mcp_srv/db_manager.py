import sys
from os.path import dirname, realpath

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = dirname(dirname(dirname(realpath(__file__))))
sys.path.append(PROJECT_ROOT)

from app.core.setting import settings
from app.core.db_access import DatabaseSchema, DatabaseSearchResult, Database

mcp = FastMCP("db_manager", port=8101)


@mcp.tool(structured_output=True)
async def get_database(hint: str = None, topN: int = 1) -> DatabaseSearchResult:
    """Search for number of 'topN' databases and tables schemas related to 'hint'."""
    # TODO: leverage semantic seach + BM25 + reranker
    return DatabaseSearchResult(
        search_result=[
            DatabaseSchema(
                database=settings.DATABASE,
                table_schema=settings.DB_SCHEMA
            )
        ],
    )


@mcp.tool()
async def query_database(database: str, sql: str) -> str:
    """Execute SQL script to retrieve data from database"""
    db = Database(**settings.DB_CONNECTION)
    return db.query_database(database=database, sql=sql)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
