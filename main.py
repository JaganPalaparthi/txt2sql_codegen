# app.py
import os, sqlite3, io, csv, uuid, tempfile
from datetime import date
from typing import List
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import openai
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "financial_data.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="NL‑SQL Financial Query Service")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "change_this_secret"),
    same_site="lax",
    max_age=30 * 60,
)

class AskPayload(BaseModel):
    question: str

GREETING_KEYWORDS = ("hi", "hello", "hey", "good morning", "good evening", "thanks")

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return any(t == kw or t.startswith(f"{kw} ") for kw in GREETING_KEYWORDS)

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def get_schema(table_name: str = "financial_data") -> str:
    conn = get_conn(); cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    schema = "\n".join(f"- {c[1]} {c[2]}" for c in cur.fetchall())
    conn.close()
    return schema

def paraphrase(history: List[str], new_question: str) -> str:
    messages = [
        {"role": "system", "content":
            "You are a paraphrasing assistant. Given past user questions, rewrite only the latest one briefly without altering intent."}
    ]
    for q in history[-5:]:
        messages.append({"role": "assistant", "content": f"(earlier) {q}"})
    messages.append({"role": "user", "content": new_question})
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def generate_sql(paraphrased: str, schema: str) -> str:
    system_msg = (
        "You are a SQL assistant. The only table is `financial_data` with schema:\n"
        f"{schema}\n\n"
        "If this is a greeting (e.g. “hi”, “hello”), return exactly `--GREETING`. "
        "Otherwise generate exactly one SELECT statement. If you cannot answer, output exactly `--UNANSWERABLE`."
    )
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": paraphrased},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def analyze_results(sql: str, cols: List[str], rows: List[List]) -> str:
    rows_sample = [dict(zip(cols, row)) for row in rows[:5]]
    prompt = (
        "You are a data analyst. Summarize the SQL results: trends, anomalies, averages, top/bottom rows.\n"
        f"SQL:\n{sql}\nColumns: {cols}\nSample rows: {rows_sample}\n"
        "Keep it concise (≤150 words)."
    )
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

@app.post("/ask")
async def ask(payload: AskPayload, request: Request, background_tasks: BackgroundTasks):
    q = payload.question.strip()
    session = request.session

    if is_greeting(q):
        session.clear()
        return JSONResponse({"reply": "Hello there! 👋 How can I help with your financial data?"})

    history = session.get("history", [])
    history.append(q)
    session["history"] = history

    try:
        paraphrased = paraphrase(history[:-1], q)
    except Exception as e:
        session.clear()
        raise HTTPException(status_code=500, detail=f"Paraphrasing failed: {e}")

    schema = get_schema()
    try:
        sql = generate_sql(paraphrased, schema)
    except Exception as e:
        session.clear()
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {e}")

    if sql.strip() == "--GREETING":
        session.clear()
        return JSONResponse({"paraphrased": paraphrased, "reply": "Hi again! 👋"})

    if sql.upper().startswith("--UNANSWERABLE"):
        session.clear()
        return JSONResponse({"paraphrased": paraphrased, "reply": "Sorry, I can’t answer that using the available data."})

    # Execute SQL
    try:
        conn = get_conn(); cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
    except sqlite3.Error as e:
        session.clear()
        raise HTTPException(status_code=400, detail=f"SQLite error: {e}")

    analysis = analyze_results(sql, cols, rows) if rows else "No matching rows to analyze."

    response_payload = {
        "paraphrased": paraphrased,
        "sql": sql,
        "results": [dict(zip(cols, r)) for r in rows],
        "analysis": analysis,
    }

    if rows:
        file_id = uuid.uuid4().hex
        path = os.path.join(tempfile.gettempdir(), f"{file_id}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        background_tasks.add_task(lambda p: os.remove(p), path)
        session.clear()
        response_payload["download_url"] = f"/download/{file_id}"

    return JSONResponse(jsonable_encoder(response_payload))

@app.get("/download/{file_id}")
def download(file_id: str):
    path = os.path.join(tempfile.gettempdir(), f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="text/csv", filename="results.csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
