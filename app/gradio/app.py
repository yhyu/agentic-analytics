import warnings
warnings.filterwarnings("ignore")

import os
import gradio as gr
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.agent import Agent
from app.core.constant import *
from app.core.llm import LLM
from app.core.setting import logger

app = FastAPI()
llms = LLM.get_llms()
agent = Agent(llms)


def new_report(states):
    states['tid'] = None
    return datetime.now(), states, None,


def stop_gen(states):
    logger.info('stop button is pressed.')
    agent.set_stop_flag(states.get('tid'))


def download_report(content, states):
    if not content:
        return None
    file_name = f'{states.get('tid', 'unknown_tid')}_{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.pdf'
    logger.info(f'download_report: {file_name}')
    return Agent.save_pdf(file_name, content)


async def submit_question(question, history, now: datetime, states: dict):
    tid, response, ret_code = await agent.run_analysis(question, states.get('tid'), now, 60)
    states['tid'] = tid
    if ret_code == ERR_DONE:
        return ['Report generated', response, gr.update(placeholder='Type a follow up request...'), states]
    else:
        if ret_code == ERR_QUESTION:
            placeholder = 'Answer the question...'
        else:
            placeholder = 'Type a request...'
        return [response, None, gr.update(placeholder=placeholder), states]

with gr.Blocks(
    title='Deep Analytics', theme=gr.themes.Base(), css='.progress-text { display: none !important; }',
    fill_height=True, fill_width=True,
) as gr_app:
    states = gr.State({})
    report_box = gr.Markdown(
        label='Report',
        min_height=516,
        max_height=516,
        container=True,
        show_copy_button=True,
        render=False,
        padding=True,
        elem_id='report_markdown'
    )
    chatbot_textbox = gr.Textbox(
        placeholder='Type a request...',
        show_label=False,
        label="",
        autofocus=True,
        submit_btn=True,
        stop_btn=True,
    )

    with gr.Row(equal_height=False):
        with gr.Column():
            date_component = gr.DateTime(
                label='Current Time',
                type='datetime',
                value=datetime.now(),
                elem_id='date_component',
                scale=0,
            )
            input_bar = gr.ChatInterface(
                fn=submit_question,
                additional_outputs=[report_box, chatbot_textbox, states],
                type='messages',
                additional_inputs=[date_component, states],
                textbox=chatbot_textbox,
                fill_height=True,
                fill_width=True,
            )
        with gr.Column():
            report_box.render()
            with gr.Row(scale=0):
                download_btn = gr.DownloadButton(
                    label='Download Report (pdf)',
                    visible=True,
                    interactive=False,
                    elem_id='download_btn'
                )
                clear_button = gr.ClearButton(
                    value='New Report',
                    components=[input_bar.textbox, input_bar.chatbot, report_box, download_btn]
                )

    input_bar.textbox.stop(fn=stop_gen, inputs=states)
    chatbot_textbox.submit(show_progress='hidden')
    report_box.change(
        fn=download_report,
        inputs=[report_box, states],
        outputs=download_btn
    ).then(
        fn=lambda content: gr.update(interactive=(content and len(content) > 0)),
        inputs=[download_btn],
        outputs=download_btn,
    )
    clear_button.click(
        fn=new_report,
        inputs=states,
        outputs=[date_component, states, download_btn]
    ).then(
        fn=lambda: gr.update(interactive=False),
        outputs=download_btn,
    )

os.makedirs("charts", exist_ok=True)
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

app = gr.mount_gradio_app(app, gr_app, path="/app")
