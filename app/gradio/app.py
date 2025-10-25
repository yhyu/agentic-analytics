import asyncio
import os
import gradio as gr
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.agent import Agent
from app.core.constant import *
from app.core.llm import LLM
from app.core.setting import logger

app = FastAPI()
llms = LLM.get_llms()
agent = asyncio.run(Agent.create(llms))


def new_report(states):
    states['tid'] = None
    return datetime.now(), states, None,


def stop_gen(states):
    logger.info('stop button is pressed.')
    agent.set_stop_flag(states.get('tid'))


def download_report(content, states):
    if not content:
        return None
    return agent.save_pdf(states.get('uid'), states.get('tid'), content)


def gen_report_links(uid: str, reports: list[tuple[str, str]]) -> list[str]:
    links = []
    for r in reports:
        tid = r[0]
        base_name = r[1].rstrip('.pdf')
        elements = base_name.rsplit('_', 1)
        if len(elements) != 2:
            continue
        report_time = datetime.strftime(datetime.strptime(elements[1], '%Y%m%d%H%M%S'), '%Y-%m-%d %H:%M:%S')
        href = f'document.location.pathname="reports/{uid}/{tid}/{base_name}";'
        links.append(f"<div><a href='javascript:;' onclick='{href}'>{elements[0]} [{report_time}]</a></div>")
    return links


def sidebar_expand(states):
    uid = states.get('uid', 'anonym')
    tid = states.get('tid')
    session_reports, user_reports = agent.list_reports(uid, tid)
    return ''.join(gen_report_links(uid, session_reports)), ''.join(gen_report_links(uid, user_reports))


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
    with gr.Sidebar(open=False) as side_bar:
        gr.HTML('<h3>Reports</h3>')
        with gr.Tab('Current Session'):
            session_content = gr.HTML()
        with gr.Tab('History'):
            history_content = gr.HTML()

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
    side_bar.expand(
        fn=sidebar_expand,
        inputs=states,
        outputs=[session_content, history_content],
    )


@app.get('/reports/{uid}/{tid}/{filename}')
async def download_hist_report(uid: str, tid: str, filename: str):
    target = f'reports/{uid}/{tid}/{filename}.pdf'
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(target, media_type="application/octet-stream", filename=filename)

os.makedirs("charts", exist_ok=True)
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

app = gr.mount_gradio_app(app, gr_app, path="/app")
