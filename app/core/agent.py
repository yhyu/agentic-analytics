import json
import operator
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from typing import TypedDict, Annotated, Any, Literal, Union, List
from uuid import uuid4 as guid

from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from markdown_pdf import MarkdownPdf, Section

import app.core.utils as utils
from app.core.constant import *
from app.core.db_access import DBAccess, DBLookUp
from app.core.llm import LLM
from app.core.setting import settings, logger
from app.core.utils import Singleton

CLEAN = sys.intern('_clean_')


def clean_reducer(current_value: list, new_value: list):
    if CLEAN in new_value:
        return []
    return current_value + new_value


class InputValidation(BaseModel):
    description: str = Field(description="Describe why you're accept or reject.")
    accepted: bool = Field(description="Does the request belongs to the topic that you accept.")


class ActionResult(TypedDict):
    task_id: int
    purpose: str
    result: Any


class ActionCode(BaseModel):
    code: str = Field(description="either sql query script, or python code, or web search query")


class ActionConfig(TypedDict):
    task_id: Annotated[int, "the is of the action."]
    purpose: Annotated[str, "the purpose of the action"]
    action_type: Annotated[Literal['sql', 'python', "web search"], "Type of action."]
    database: Annotated[str, "sqlite database file name, or mysql and PostgreSQL database name."]
    code: Annotated[str, "either sql query script, or python code, or web search query"]
    dependencies: Annotated[list[int], "list of dependent action's task_id"]


class ActionList(TypedDict):
    actions: Annotated[List[ActionConfig], "list of action"]


class ActionState(TypedDict):
    action: ActionConfig
    data: Annotated[list[Any], operator.add]
    finished_actions: Annotated[List[ActionResult], clean_reducer]
    intent: str
    error: str
    success: bool


class PyCodeState(ActionState):
    shell_cmd: str = None


class ShellCommand(BaseModel):
    shell_cmd: str = Field(description='shell command')


class AgentState(TypedDict):
    background: str = ''  # compressed history
    start_index: int = 0  # start message index of curent run
    data_index: int = 0   # start data index of curent run
    requests: Annotated[list[str], operator.add]  # original requests
    intents: Annotated[list[str], operator.add]   # real intents
    reports: Annotated[list[str], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]
    done: bool = False
    plans: Annotated[list[str], operator.add]
    actions: ActionList
    finished_actions: Annotated[List[ActionResult], clean_reducer]
    data: Annotated[list[Any], operator.add]
    invalid_request: str = None


class AskHumanInput(BaseModel):
    question: str = Field(description='A question to ask the human assistant.')


class Agent(metaclass=Singleton):

    def __init__(self, llms):
        self._lock = threading.Lock()
        self._stop_flag = dict()  # {tid: stop_flag}
        self.llm = llms['Flash']
        self.llm_cot = llms['Thinking']  # reasoning model, or llm with CoT
        self.human_tools = [Agent.human_assistant]
        self.memory = MemorySaver()
        self.graph = self.build_graph(self.memory)
        self.db_access = DBAccess(**settings.DB_CONNECTION)
        self.get_database = DBLookUp().get_database

    @staticmethod
    @tool('human-assistant-tool', args_schema=AskHumanInput, return_direct=True)
    def human_assistant(question: str):
        """Human assistant answer any question"""
        response = interrupt({'question': question})
        return response['answer']

    @staticmethod
    def save_pdf(filename: str, content: str):
        os.makedirs("reports", exist_ok=True)
        file_path = os.path.join('reports', filename)
        pdf = MarkdownPdf()
        content = content.replace(settings.HOST_URL, './')
        pdf.add_section(Section(content, toc=False), user_css=settings.REPORT_PDF_CSS)
        pdf.save(file_path)
        return file_path

    def query_database(self, database: str, sql: str):
        return self.db_access.query_database(database, sql)

    # Agent Nodes

    def _starter(self, state: AgentState):
        return {
            "requests": [state['messages'][-1].content],
            "start_index": len(state['messages']) - 1,
        }

    def _topic_validator(self, state: AgentState):
        prompts = LLM.Prompts['topic_validator']
        system_prompt = prompts['system'].format(topics=settings.ACCEPTED_TOPICS)
        if state.get("background"):
            system_prompt += f"\nHere are some background:{state["background"]}"
        result = self.llm.with_structured_output(InputValidation).invoke(
            [SystemMessage(content=system_prompt)] +
            state["messages"][state.get("start_index", 0):] +
            [HumanMessage(content=prompts['user'])]
        )
        if result.accepted:
            return {'invalid_request': None}
        else:
            logger.info(f"user request is not accepted: '{result.description}'")
            return {'invalid_request': result.description}

    async def _request_clarifier(self, state: AgentState):
        db_tables = await self.get_database(state['messages'][state.get("start_index", 0)].content)
        prompts = LLM.Prompts['clarifier']
        system_prompt = prompts['system']
        if state.get("background"):
            system_prompt += f"\nHere are some background:{state["background"]}"
        db_schemas = '\n---\n'.join(
            [
                f"db type: {self.db_access.db_type}\ndatabase: {db}\ntable schema:\n{tbl}"
                for db, tbl in db_tables
            ]
        )
        result = await self.llm_cot.bind_tools(self.human_tools).ainvoke(
            [SystemMessage(content=system_prompt)] +
            state["messages"][state.get("start_index", 0):] +
            [HumanMessage(content=prompts['user'].format(db_schemas=db_schemas))]
        )
        return {"messages": [result]}

    async def _planner(self, state: AgentState):
        intents = state['messages'][-1].content
        db_tables = await self.get_database(intents)
        prompts = LLM.Prompts['planner']
        system_prompt = prompts['system']
        if state.get("background"):
            system_prompt += f"\nHere are some background:{state["background"]}"
        db_schemas = '\n---\n'.join(
            [
                f"db type: {self.db_access.db_type}\ndatabase: {db}\ntable schema:\n{tbl}"
                for db, tbl in db_tables]
        )
        return {
            "plans": [
                (await self.llm_cot.ainvoke(
                    [SystemMessage(content=system_prompt)] +
                    [HumanMessage(content=prompts['user'].format(user_intents=intents, db_schemas=db_schemas))]
                )).content
            ],
            "intents": [intents],
            "finished_actions": [CLEAN]
        }

    async def _action_extractor(self, state: AgentState):
        db_tables = await self.get_database(state['intents'][-1])
        prompts = LLM.Prompts['action_exactor']
        system_prompt = prompts['system']
        if len(state["reports"]) > 0:
            system_prompt += f"\nHere is the previous report:\n{state["reports"][-1]}"
        db_schemas = '\n---\n'.join(
            [
                f"db type: {self.db_access.db_type}\ndatabase: {db}\ntable schema:\n{tbl}"
                for db, tbl in db_tables]
        )
        user_prompt = prompts['user'].format(db_schemas=db_schemas, plan=state["plans"][-1])
        if state.get('finished_actions'):
            user_prompt += (f"Based on above analytics plan and following finished tasks, if there is unfinished tasks, extract the tasks, "
                            f"otherwise, output empty actions.\n"
                            f"Here are finished tasks:\n{'\n'.join([json.dumps(a, ensure_ascii=False) for a in state.get('finished_actions')])}"
                            f"\n\nNotes: DO NOT output redundant tasks, and DO NOT output tasks that are already in the finieshed tasks.")
        response = await self.llm.with_structured_output(ActionList).ainvoke(
            [
                SystemMessage(content=prompts['system']),
                HumanMessage(content=user_prompt),
            ]
        )
        logger.info(f"action_extractor extracted {len(response.get("actions"))} actions in this iteration.")
        # remove tasks that have dependencies
        finished_action_ids = set([a['task_id'] for a in state.get('finished_actions', [])])
        actions = []
        for a in response.get("actions", []):
            pending = False
            for d in a.get('dependencies', []):
                if d not in finished_action_ids:
                    pending = True
                    break
            if not pending:
                actions.append(a)
        logger.info(f"action_extractor extracted {len(actions)} actions that have no dependencies in this iteration.")
        return {"actions": actions}

    def _action_distributor(self, state: AgentState):
        return {}

    async def _data_collector(self, state: ActionState):
        if state.get('error', '') == '':
            return {}
        db_tables = await self.get_database(state['intent'])
        db_schemas = '\n---\n'.join(
            [
                f"db type: {self.db_access.db_type}\ndatabase: {db}\ntable schema:\n{tbl}"
                for db, tbl in db_tables
            ]
        )
        code_types = {
            'python': 'python code',
            'sql': 'sql script',
            'web search': 'web search'
        }
        display_code_type = code_types.get(state['action']['action_type'], 'code')
        prompt = (
            f"You are a senior software engineer and data scientist, good at debug. The {display_code_type} has following error. "
            "Please refine and fixed the error, and output the correct {display_code_type} in the 'code' field.\n"
            "Here is related database tables.\n"
            f"{db_schemas}\n---\n\n"
        )
        if 'purpose' in state['action']:
            prompt += f"purpose: {state['action']['purpose']}\n"
        if state['action']['action_type'] == 'sql':
            prompt += f"sql: {state['action']['code']}\n"
        elif state['action']['action_type'] == 'python':
            prompt += f"python code: {state['action']['code']}\n"
        else:
            prompt += f"web search: {state['action']['code']}\n"
        prompt += f"error: {state['error']}"
        if state['action']['action_type'] == 'python':
            prompt += "\nPlease be sure the visualizations/charts are readable, DO NOT clutter it up in a single visualization/chart."
        response = await self.llm.with_structured_output(ActionCode).ainvoke(
            [HumanMessage(content=prompt)]
        )
        action = state['action']
        action['code'] = response.code
        return {"action": action}

    def _report_writer(self, state: AgentState):
        prompts = LLM.Prompts['reporter']
        system_prompt = prompts['system']
        if len(state["reports"]) > 0:
            system_prompt += f"\nHere is the previous report:\n{state["reports"][-1]}"
        ai_message = self.llm_cot.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=prompts['user'].format(
                        user_intents=state['intents'][-1],
                        plan=state['plans'][-1],
                        data='\n\n'.join(state['data'][state.get("data_index", 0):]), endpoint=settings.HOST_URL)
                ),
            ]
        )
        background = f"User Intents:\n{'\n'.join(state["intents"])}\n---\nPrevious Report:\n{ai_message.content}\n---\n"
        return {
            "messages": [AIMessage(content='Report generated')],
            "reports": [ai_message.content],
            "background": background,
            "data_index": len(state['data']) - 1,
        }

    def _web_search(self, state: ActionState):
        if 'code' not in state['action']:
            return {}
        search_result = utils.web_search(state['action']['code'])
        if search_result.startswith('Error'):
            return {'error': search_result}
        # summarize result
        prompts = LLM.Prompts['web_searcher']
        result = self.llm.invoke(
            [
                SystemMessage(content=prompts['system']),
                HumanMessage(
                    content=prompts['user'].format(
                        purpose=state['action'].get('purpose', ''), search_results=search_result)
                )
            ]
        ).content
        return {
            'data': [f"*{state['action'].get('purpose', '')}\n{result}"],
            'finished_actions': [{
                'task_id': state['action'].get('task_id', ''),
                'purpose': state['action'].get('purpose', ''),
                'result': result,
            }],
            'error': '', 'success': True
        }

    def _run_pycode(self, state: PyCodeState):
        try:
            with tempfile.TemporaryFile() as f_out, tempfile.TemporaryFile() as f_err:
                subprocess.run(
                    ["python3", "-c", 'import warnings\nwarnings.filterwarnings("ignore")\n' + state['action']['code']],
                    stdout=f_out,
                    stderr=f_err,
                )
                f_out.seek(0)
                f_err.seek(0)
                output = ''.join([line.decode("utf-8") for line in f_out.readlines()])
                error = '\n'.join([line.decode("utf-8") for line in f_err.readlines()])
                if error:
                    return {'error': error}
                elif output:
                    return {
                        'data': [f"*{state['action'].get('purpose', '')}\n{output}"],
                        'finished_actions': [{
                            'task_id': state['action'].get('task_id', ''),
                            'purpose': state['action'].get('purpose', ''),
                            'result': output,
                        }],
                        'error': '', 'success': True
                    }
                else:
                    return {'error': 'output files are not generated.'}
        except Exception:
            error = traceback.format_exc()
            return {'error': error}

    def _run_shell(self, state: PyCodeState):
        subprocess.run(
            state['shell_cmd'],
            shell=True,
            capture_output=True,
            text=True,
        )
        return {'shell_cmd': None}  # reset command

    async def _run_sql(self, state: ActionState):
        try:
            result = self.query_database(state['action'].get('database'), state['action']['code'])
            if result.startswith('Error'):
                return {'error': result}
            return {
                'data': [f"*{state['action'].get('purpose', '')}\n{result}"],
                'finished_actions': [{
                    'task_id': state['action'].get('task_id', ''),
                    'purpose': state['action'].get('purpose', ''),
                    'result': result,
                }],
                'error': '', 'success': True
            }
        except Exception:
            error = traceback.format_exc()
            return {'error': error}

    def _pycode_debuger(self, state: PyCodeState):
        if state.get('success'):
            return {}
        prompts = LLM.Prompts['pycode_debuger']
        result = self.llm.with_structured_output(ShellCommand).invoke(
            [
                SystemMessage(content=prompts['system']),
                HumanMessage(
                    content=prompts['user'].format(
                        python_code=state['action']['code'], error=state.get('error'),
                    ))
            ]
        )
        if result.shell_cmd and len(result.shell_cmd) > 0:
            return {'shell_cmd': result.shell_cmd}
        else:
            return {}

    def _run_shell_condition(self, state: PyCodeState):
        if state.get('shell_cmd'):
            return 'shell'
        else:
            return 'next'

    def _data_collection_condition(self, state: ActionState):
        if state.get('success'):
            return 'next'
        return state['action']['action_type']

    def _my_tools_condition(
            self,
            state: Union[list[AnyMessage], dict[str, Any], BaseModel],
            messages_key: str = "messages",
    ):
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
            ai_message = messages[-1]
        elif messages := getattr(state, messages_key, []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "next"

    def _continue_data_collection_condition(self, state: AgentState):
        return [Send("data_collector", {"action": a, "intent": state["intents"][-1]}) for a in state["actions"]]

    def build_graph(self, memory):
        # run python code subgraph
        pycode_flow = StateGraph(PyCodeState)
        pycode_flow.add_node('run_pycode', self._run_pycode)
        pycode_flow.add_node('pycode_debuger', self._pycode_debuger)
        pycode_flow.add_node('run_shell', self._run_shell)
        pycode_flow.add_edge('run_pycode', 'pycode_debuger')
        pycode_flow.add_edge('run_shell', 'run_pycode')
        pycode_flow.add_conditional_edges(
            'pycode_debuger',
            self._run_shell_condition,
            {'shell': 'run_shell', 'next': END}
        )
        pycode_flow.set_entry_point('run_pycode')
        pycode_graph = pycode_flow.compile(checkpointer=memory)

        # Action subgraph
        act_flow = StateGraph(ActionState)
        act_flow.add_node('data_collector', self._data_collector)
        act_flow.add_node('web_search', self._web_search)
        act_flow.add_node('run_sql', self._run_sql)
        act_flow.add_node('run_code', pycode_graph)

        act_flow.set_entry_point('data_collector')
        act_flow.add_edge('run_sql', 'data_collector')
        act_flow.add_edge('run_code', 'data_collector')
        act_flow.add_edge('web_search', 'data_collector')
        act_flow.add_conditional_edges(
            'data_collector',
            self._data_collection_condition,
            {'next': END, 'sql': 'run_sql', 'python': 'run_code', 'web search': 'web_search'}
        )
        act_graph = act_flow.compile(checkpointer=memory)

        # main graph
        workflow = StateGraph(AgentState)
        workflow.add_node('starter', self._starter)
        workflow.add_node('topic_validator', self._topic_validator)
        workflow.add_node('request_clarifier', self._request_clarifier)
        workflow.add_node('planner', self._planner)
        workflow.add_node('action_extractor', self._action_extractor)
        workflow.add_node('action_distributor', self._action_distributor)
        workflow.add_node('data_collector', act_graph)
        workflow.add_node('report_writer', self._report_writer)
        workflow.add_node('human_assistant', ToolNode(self.human_tools))

        workflow.add_edge(START, 'starter')
        workflow.add_edge('starter', 'topic_validator')
        workflow.add_conditional_edges(
            'topic_validator',
            lambda state: 'end' if state.get('invalid_request') else 'continue',
            {'end': END, 'continue': 'request_clarifier'}
        )
        workflow.add_conditional_edges(
            'request_clarifier',
            self._my_tools_condition,
            {'tools': 'human_assistant', 'next': 'planner'}
        )
        workflow.add_edge('human_assistant', 'topic_validator')
        workflow.add_edge('planner', 'action_extractor')
        workflow.add_conditional_edges(
            'action_extractor',
            lambda state: 'continue' if state.get('actions') else 'next',
            {'continue': 'action_distributor', 'next': 'report_writer'}
        )
        workflow.add_conditional_edges(
            'action_distributor',
            self._continue_data_collection_condition,
            ["data_collector"]
        )
        workflow.add_edge('data_collector', 'action_extractor')
        workflow.add_edge('report_writer', END)
        return workflow.compile(checkpointer=memory)

    def set_stop_flag(self, tid: str = None, flag: bool = True) -> bool:
        with self._lock:
            if not tid:
                return False
            self._stop_flag[tid] = flag
            return True

    def get_stop_flag(self, tid: str = None) -> bool:
        with self._lock:
            if not tid:
                return False
            return self._stop_flag.get(tid, False)

    async def run_analysis(self, command: str, tid: str = None, now: datetime = datetime.now(), recursion_limit: int = 25):
        logger.info(f'run_analysis: {command} (tid: {tid})')
        response = ""
        if tid:
            config = {"configurable": {"thread_id": tid}, "recursion_limit": recursion_limit}
            current_state = self.graph.get_state(config)
            next_step = current_state.next
            if not next_step:
                command = f"Current time is {now.strftime('%c')}.\n{command}"
            elif isinstance(next_step, (tuple, list)) and len(next_step) > 0:
                next_step = next_step[0]
            if next_step == 'human_assistant':  # next is tool call
                messages = Command(resume={"answer": command})
            else:
                messages = {"messages": [HumanMessage(content=command)]}
        else:
            tid = str(guid().hex)
            config = {"configurable": {"thread_id": tid}, "recursion_limit": recursion_limit}
            messages = {"messages": [HumanMessage(content=f"Current time is {now.strftime('%c')}.\n{command}")]}
        async for s in self.graph.astream(messages, config, subgraphs=True):
            logger.info(s)
            logger.info('next step: ' + ', '.join(self.graph.get_state(config).next))
            if self.get_stop_flag(tid):
                self.set_stop_flag(tid, False)  # reset
                logger.info('user canceled generation.')
                break
            if isinstance(s, dict):
                s = (s)
            for d in s:
                if '__interrupt__' in d:
                    response = d['__interrupt__'][0].value.get('question')
                    break
        current_state = self.graph.get_state(config)
        if current_state.values.get('invalid_request'):
            return tid, current_state.values.get('invalid_request'), ERR_INVALID
        ret_code = ERR_QUESTION
        done = not current_state.next
        if done:
            response = current_state.values['reports'][-1]
            ret_code = ERR_DONE
            logger.info(f"report generated (tid: {tid})")
        return tid, response, ret_code
