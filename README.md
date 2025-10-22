# Agentic Analytics
Agentic Analytics is a project that leverages Large Language Models (LLMs) to empower non-technical users, such as sales and product managers, to easily obtain business insights and detailed analysis reports by asking questions in natural language. For example, a sales team could simply ask, '*Generate a monthly report based on our sales data from the previous month.*'

## Overview
![Overview](https://github.com/yhyu/agentic-analytics/blob/main/images/overview.png)  
- **Input Guardrail** is a simple guardrail to limit user's requirement are within the specific topics. You can change supported topics by setting **ACCEPTED_TOPICS** environment variable.  
- **User Intent Confirmation** is to clarify user requests. The user can simply approve it or describe the request in more detail.  
- **Planning** is to plan all tasks to achieve user requests.  
- **Task Distribution** schedules tasks in parallel based on their dependencies.  
- **Task Execution** executes each task independently.  
- **Summary** writes the final report. (You can customize the report pdf style by editing the **REPORT_PDF_CSS** environment variable.)  

## Sample dataset for demo
- To demostrate the project, [Warehouse and Retail Sales](https://catalog.data.gov/dataset/warehouse-and-retail-sales) dataset is used. (The sample dataset is between 2017/06 and 2020/09, so you have to adjust the current time in the upper left component.)
- You can replace with your dataset by setting **DATABASE** and **DB_SCHEMA** environment variables, or use **MCP** (refer to [sample MCP server](https://github.com/yhyu/agentic-analytics/blob/main/app/mcp_srv/db_searcher.py).)
  
## Quick start
Create python virtual environment.
```bash
python -m venv agentic_analytics
source agentic_analytics/bin/activate
```

Setup environment variables.  
Environment variables are divided into two env files (of course you can put them together), LLM service-related variables (*.env*) and others (*common.env*). For example, for OpenAI, copy openai.env to .env and modify your api key and models.
```bash
cp openai.env .env
```

Install dependent packages.
```bash
pip install --no-cache-dir -r requirements.txt
```

Download sample data (or use your own data.)
```bash
python download_sample_db.py
```

Start service
```bash
gunicorn app.gradio.app:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

## Try it out
1. Open http://127.0.0.1:8000/app
  ![Gradio interface](https://github.com/yhyu/agentic-analytics/blob/main/images/gradio_ui.png)
   - All conversations are memorized untill you click the **New Report** button or refresh your browser.
2. Make a request
  ![Confirm request](https://github.com/yhyu/agentic-analytics/blob/main/images/confirm.png)
  ![Make a request](https://github.com/yhyu/agentic-analytics/blob/main/images/main_request.png)
3. Subsequent request
  ![Subsequent request](https://github.com/yhyu/agentic-analytics/blob/main/images/followup_request.png)
4. Session and history reports (in theleft sidebar)
  ![Session and History reports](https://github.com/yhyu/agentic-analytics/blob/main/images/history_reports.png)
5. Try different LLMs.
   - You can select different LLMs by changing the **LLM_FLASH_MODEL** and **LLM_THINKING_MODEL** environment variables.

## Run in Docker
You can also run the project in Docker.

Build docker image
```bash
docker build -f Dockerfile.gradio -t agentic-analytics:latest .
```

Start docker
```bash
docker compose -f docker-compose.gradio.yml up -d
```
The first time may take several seconds due to sample dataset preparation.

## MCP support
In order to support customized algorithms for semantic relevant database searching as well as data warehose access, the solution also supports MCP. You can enable MCP support by uncommenting and setting DB_ACCESS_MCP_XXX and/or DB_SEARCH_MCP_XXX environments variables.  
(To try out the [MCP server example](https://github.com/yhyu/agentic-analytics/blob/main/app/mcp_srv/db_manager.py), you can execute the following command.)  
```bash
cd app/mcp_srv
python db_manager.py
```

## Agentic workflow
As you can see the following workflow, it's different from other famous deep agent implementations, such as [LangChain Deep Agents](https://blog.langchain.com/deep-agents/), in that it's not autonomous like others, instead, it's more deterministic. This's becuase I want the solution to still be effective even with SLM, especially in an on-premises environment.  
![workflow](https://github.com/yhyu/agentic-analytics/blob/main/images/agent_flow.png)
