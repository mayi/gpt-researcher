from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
from backend.websocket_manager import WebSocketManager
from backend.utils import write_md_to_pdf, write_md_to_word
from contextlib import asynccontextmanager


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str

# Dynamic directory for outputs once first research is run
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")

templates = Jinja2Templates(directory="./frontend")

manager = WebSocketManager()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "report": None})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start"):
                json_data = json.loads(data[6:])
                task = json_data.get("task")
                report_type = json_data.get("report_type")
                if task and report_type:
                    report = await manager.start_streaming(task, report_type, websocket)
                    if report:
                        print("Report:", report)
                        # Saving report as pdf
                        pdf_path = await write_md_to_pdf(report)
                        # Saving report as docx
                        docx_path = await write_md_to_word(report)
                        # Returning the path of saved report files
                        await websocket.send_json({"type": "path", "output": {"pdf": pdf_path, "docx": docx_path}})
                    else:
                        print("Error: report not generated.")
                else:
                    print("Error: not enough parameters provided.")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)

