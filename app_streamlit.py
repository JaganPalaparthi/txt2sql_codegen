import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO

API_URL = os.environ.get("DATA_QUERY_API_URL", "http://127.0.0.1:8000/ask")

st.set_page_config(page_title="Data Query", layout="wide")
st.title("💬 Data Query")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def call_api(question: str) -> dict:
    resp = requests.post(API_URL, json={"question": question}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Safely cast download URL to string
    cd = data.get("csv_download_url")
    if cd is not None:
        cd = str(cd)
        if cd.startswith("/"):
            base = API_URL.split("/ask")[0]
            cd = base + cd
        try:
            csv_resp = requests.get(cd, timeout=120)
            csv_resp.raise_for_status()
            data["_csv_bytes"] = csv_resp.content
        except Exception:
            pass

    vid = data.get("viz_id")
    if vid:
        viz_url = API_URL.rstrip("/ask") + f"/visualize/{vid}"
        vz = requests.get(viz_url, timeout=120)
        vz.raise_for_status()
        data["viz_html"] = vz.json().get("html")

    return data

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(str(msg["content"]))
        if df := msg.get("data"):
            st.dataframe(df)
            if b := msg.get("_csv_bytes"):
                st.download_button("💾 Download CSV", data=b, file_name="results.csv", mime="text/csv")
        if viz_html := msg.get("viz_html"):
            st.subheader("📊 Interactive Chart")
            components.html(viz_html, height=450, scrolling=True)
        if analysis := msg.get("analysis"):
            with st.expander("🧠 Show explanation"):
                st.write(analysis)

if question := st.chat_input("Enter your question…"):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        res = call_api(question)
    except Exception as e:
        err = f"⚠️ Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    else:
        reply = res.get("reply") or ("Here are your results:" if res.get("results") else "No data found.")
        entry = {"role": "assistant", "content": reply}

        if rows := res.get("results"):
            df = pd.DataFrame(rows)
            entry["data"] = df
        if b := res.get("_csv_bytes"):
            entry["_csv_bytes"] = b
        if a := res.get("analysis"):
            entry["analysis"] = a
        if h := res.get("viz_html"):
            entry["viz_html"] = h

        st.session_state.messages.append(entry)

        with st.chat_message("assistant"):
            st.write(reply)
            if "data" in entry:
                st.dataframe(entry["data"])
            if b := entry.get("_csv_bytes"):
                st.download_button("💾 Download CSV", data=b, file_name="results.csv", mime="text/csv")
            if h := entry.get("viz_html"):
                st.subheader("📊 Interactive Chart")
                components.html(h, height=450, scrolling=True)
            if a := entry.get("analysis"):
                with st.expander("🧠 Show explanation"):
                    st.write(a)
