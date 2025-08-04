python3 -m venv txt2sql-env
 .\txt2sql-env\Scripts\activate
 python3 data_generation_syn.py
 fastapi
uvicorn[standard]
python-dotenv
openai
pandas
matplotlib
mpld3
plotly
streamlit
requests

python -m venv venv
venv\Scripts\activate

OPENAI_API_KEY: your OpenAI secret key

SQLITE_DB_PATH: path to financial_data.db (default is financial_data.db)

SESSION_SECRET: any random secret (for session cookies)

uvicorn main1:app --host 127.0.0.1 --port 8000 --reload

streamlit run app_streamlit.py

