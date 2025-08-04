# main1.py
import os
import sqlite3
import io
import csv
import uuid
import threading
import time
import re
from typing import List, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import openai
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import plotly.express as px

load_dotenv()
DB_PATH = os.getenv("SQLITE_DB_PATH", "financial_data.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="NL‑SQL Financial Query with Visualization")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "CHANGE_ME"))

class AskPayload(BaseModel):
    question: str

_csv_store: dict = {}
_viz_store: dict = {}

def _cleanup_loop():
    while True:
        now = time.time()
        for store in (_csv_store, _viz_store):
            for key, val in list(store.items()):
                if now - val["ts"] > 900:
                    del store[key]
        time.sleep(60)

threading.Thread(target=_cleanup_loop, daemon=True).start()

def is_greeting(q: str) -> bool:
    return q.strip().lower().split()[0] in ("hi", "hello", "hey", "thanks")

def get_schema() -> str:
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        rows = conn.execute("PRAGMA table_info(financial_data)").fetchall()
    return "\n".join(f"- {r[1]} {r[2]}" for r in rows)

def llm_chat(messages: List[dict], temperature: float = 0.0):
    if openai.__version__.startswith("0."):
        return openai.ChatCompletion.create(model=OPENAI_MODEL, messages=messages, temperature=temperature)
    return openai.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=temperature)

def strip_sql(sql: str) -> str:
    fence_pat = re.compile(r"```(?:sql)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    m = fence_pat.search(sql)
    if m:
        s = m.group(1)
    else:
        s = sql
    return s.strip().rstrip(";")

def paraphrase(history: List[str], question: str) -> str:
    msgs = [{"role": "system", "content": "Paraphrase only the latest question concisely."}]
    for h in history[-5:]:
        msgs.append({"role": "assistant", "content": f"(earlier) {h}"})
    msgs.append({"role": "user", "content": question})
    resp = llm_chat(msgs, temperature=0.3)
    return resp.choices[0].message.content.strip()

def generate_sql(paraphrased: str, schema: str) -> str:
    system = (
        "You are a SQL assistant. Only table financial_data exists.\n"
        f"Schema:\n{schema}\n"
        "If the input is a greeting, return exactly --GREETING. "
        "Otherwise return one SELECT statement. If no answer, return --UNANSWERABLE."
    )
    resp = llm_chat([{"role": "system", "content": system}, {"role": "user", "content": paraphrased}], temperature=0)
    return strip_sql(resp.choices[0].message.content)

def analyze_results(sql: str, cols: List[str], rows: List[List[Any]]) -> str:
    sample = [dict(zip(cols, r)) for r in rows[:5]]
    prompt = (
        "You are a data analyst. Summarize key trends, anomalies, statistics in under 150 words.\n"
        f"SQL:\n{sql}\nColumns: {cols}\nSample rows: {sample}"
    )
    resp = llm_chat([{"role": "system", "content": prompt}], temperature=0)
    return resp.choices[0].message.content.strip()

@app.post("/ask")
async def ask(payload: AskPayload, request: Request, bg: BackgroundTasks):
    q = payload.question.strip()
    session = request.session

    if is_greeting(q):
        session.clear()
        return JSONResponse({"reply": "Hello! 👋 How can I assist you with your financial data?"})

    history = session.get("history", [])
    history.append(q)
    session["history"] = history

    parap = paraphrase(history[:-1], q)
    schema = get_schema()
    sql = generate_sql(parap, schema)

    if sql in ("--GREETING", "--UNANSWERABLE"):
        session.clear()
        return JSONResponse({"reply": "Sorry, I can't answer that with the available data."})

    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        except sqlite3.Error as err:
            session.clear()
            raise HTTPException(status_code=400, detail=f"SQLite error: {err}")

    analysis = analyze_results(sql, cols, rows) if rows else ""
    response = {"results": [dict(zip(cols, r)) for r in rows], "analysis": analysis}

    if rows:
        # Create CSV storage
        dl_id = uuid.uuid4().hex
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(cols)
        writer.writerows(rows)
        _csv_store[dl_id] = {"bytes": buf.getvalue().encode("utf-8"), "ts": time.time()}
        download_url = str(request.url_for("download_csv", download_id=dl_id))
        response["csv_download_url"] = download_url

        # Register viz for later plotting
        viz_id = uuid.uuid4().hex
        _viz_store[viz_id] = {"cols": cols, "rows": rows, "ts": time.time()}
        response["viz_id"] = viz_id

    session.clear()
    return JSONResponse(jsonable_encoder(response))

@app.get("/download/{download_id}")
async def download_csv(download_id: str):
    entry = _csv_store.pop(download_id, None)
    if not entry:
        raise HTTPException(status_code=404, detail="CSV not found")
    return StreamingResponse(
        iter([entry["bytes"]]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results-{download_id}.csv"}
    )

@app.get("/visualize/{viz_id}")
async def visualize(viz_id: str):
    entry = _viz_store.pop(viz_id, None)
    if not entry:
        raise HTTPException(status_code=404, detail="Visualization ID invalid")
    cols, rows = entry["cols"], entry["rows"]
    df = pd.DataFrame(rows, columns=cols)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        raise HTTPException(status_code=400, detail="No numeric columns to visualize")

    if len(num_cols) <= 3:
        x = cols[0]
        y_cols = num_cols
        fig, ax = plt.subplots(figsize=(6, 4))
        for y in y_cols:
            ax.plot(df[x], df[y], marker="o", linestyle="-", label=y)
        ax.set_xlabel(x); ax.legend(); ax.set_title(f"{', '.join(y_cols)} vs {x}")
        html = mpld3.fig_to_html(fig)
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import mpld3",
            "import pandas as pd",
            f"data = {rows!r}",
            f"columns = {cols!r}",
            "df = pd.DataFrame(data, columns=columns)",
            f"x = df['{x}']",
            f"y_cols = {y_cols!r}",
            "fig, ax = plt.subplots()",
            "for y in y_cols:",
            "    ax.plot(x, df[y], marker='o', label=y)",
            "ax.set_xlabel('" + x + "')",
            "ax.legend()",
            "ax.set_title('" + ", ".join(y_cols) + " vs " + x + "')",
            "html = mpld3.fig_to_html(fig)"
        ]
    else:
        fig = px.scatter_matrix(df, dimensions=num_cols[:5], title="Scatter Matrix")
        fig.update_traces(diagonal_visible=True)
        html = fig.to_html(full_html=False)
        code_lines = [
            "import pandas as pd",
            "import plotly.express as px",
            f"data = {rows!r}",
            f"columns = {cols!r}",
            "df = pd.DataFrame(data, columns=columns)",
            f"numeric = {num_cols[:5]!r}",
            "fig = px.scatter_matrix(df, dimensions=numeric)",
            "fig.update_traces(diagonal_visible=True)",
            "fig.show()"
        ]

    return JSONResponse({"html": html, "code": "\n".join(code_lines)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main1:app", host="127.0.0.1", port=8000, reload=True)
