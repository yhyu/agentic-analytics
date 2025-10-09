import logging
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=['.env', 'common.env'], env_file_encoding='utf-8')

    LLM_SERVING: str = 'openai'
    LLM_FLASH_MODEL: str = 'gpt-4.1-mini'
    LLM_THINKING_MODEL: str = 'o4-mini'
    LLM_TEMP: float = 0
    LLM_MAX_CTX: int = 40960

    SERVING_BASE_URL: Optional[str] = None
    SERVING_API_KEY: Optional[str] = None
    SERVING_MAX_RETRIES: int = 4

    # Guardrail
    # ACCEPTED_TOPICS (separated by comma)
    ACCEPTED_TOPICS: str = "sales analytics, sales report"

    HOST_URL: str = 'http://127.0.0.1:8000'

    REPORT_PDF_CSS: str = """
table {
    table-layout: fixed;
    border-collapse: collapse; /* Collapse borders into a single line */
    width: 100%; /* Make the table span the full width of its container */
    font-family: Arial, sans-serif; /* Use a clean, sans-serif font */
    border-radius: 5px; /* Optional: adds slightly rounded corners to the table */
    overflow: hidden; /* Ensures the background of the cells doesn't go outside the border-radius */
    word-wrap: break-word; /* Allows long words to break and wrap */
    overflow-wrap: break-word;
    white-space: normal;
}

/* Table header styling */
th {
    font-weight: bold; /* Make header text bold */
    font-size: 10pt;
    text-align: center; /* Align header text to the center */
    padding: 6px; /* Add padding to the header cells */
}

/* Table cell styling */
td {
    /*border: 1px solid #ddd;*/
    padding: 5px; /* Add padding to the data cells */
    border-bottom: 1px solid #ddd; /* Add a light gray bottom border to cells */
    color: #333; /* Set a dark gray text color for better readability */
    font-size: 8pt;
}"""

    # Database access
    # DB connection dict:
    # mysql: {"type": "mysql", "host": "ip address or host name", "user": "user_name", "password": "user_password", "port": 3306}
    # postgresql: {"type": "postgres", "host": "ip address or host name", "user": "user_name", "password": "user_password", "port": 5432}
    # sqlite: None
    DB_CONNECTION: Dict[str, Any] = {}

    # TODO: implement db/table selection by hybrid search
    DATABASE: Optional[str] = None
    DB_SCHEMA: Optional[str] = None

    # Use MCP
    DB_SEARCH_MCP_NAME: Optional[str] = None        # "db_searcher"
    DB_SEARCH_MCP_URL: Optional[str] = None         # "http://127.0.0.1:8001/mcp"
    DB_SEARCH_MCP_TOOL: Optional[str] = None        # "get_database"
    DB_SEARCH_MCP_TRANSPORT: Optional[str] = None   # "streamable_http"
    DB_SEARCH_MCP_HEADER: Optional[Dict[str, str]] = None  # {"Authorization": "Bearer YOUR_TOKEN"}

    # log level
    LOG_LEVEL: str = 'INFO'


# load settings
settings = AgentSettings()

logger = logging.getLogger("uvicorn.error")
logger.setLevel(settings.LOG_LEVEL)
