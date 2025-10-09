import csv
import io
import mysql.connector
import os
import psycopg2
import sqlite3
from pathlib import Path
from joblib import Memory
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.setting import settings, logger


cache_loc = os.environ.get(
    'DEEP_ANALYTICS_CACHE_LOC',
    default=os.path.join(Path.home(), 'cache_DeepAnalytics')
)
MemoryCache = Memory(location=cache_loc, verbose=0)


class DBAccess:
    def __init__(self, **kw):
        if kw and 'type' in kw:
            self.db_type = kw.pop('type')
            self.conn_args = kw
            if self.db_type == 'mysql':
                self.connector = mysql.connector
            elif self.db_type == 'postgres':
                self.connector = psycopg2
            elif self.db_type == 'sqlite':
                self.connector = sqlite3
        else:
            self.db_type = 'sqlite'
            self.connector = sqlite3
            self.conn_args = {}

    def query_database(self, database: str, sql: str):
        """Retrieve Warehouse and Retail sales data from database"""
        db_conn = None
        cursor = None
        try:
            db_conn = self.connector.connect(**(self.conn_args | {'database': database}))
            if self.db_type == 'sqlite':
                db_conn.row_factory = sqlite3.Row

            cursor = db_conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                return 'Error: no results found.'
            with io.StringIO() as f:
                w = csv.writer(f, lineterminator='\n')
                if isinstance(rows[0], dict):
                    headers = rows[0].keys()
                else:
                    headers = [h[0] for h in cursor.description]
                w.writerow(headers)
                w.writerows(rows)
                f.seek(0)
                result = f.read()

            if len(result) > 40*1024:
                return 'Error: result is more than 4K tokens. Try to aggregate of limit the size of result to make it shoter.'
            return result
        except Exception as e:
            return f"Error query database: {str(e)}"
        finally:
            if cursor:
                cursor.close()
            if db_conn:
                db_conn.close()


class DBLookUp:
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
