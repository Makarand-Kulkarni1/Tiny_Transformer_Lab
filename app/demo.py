import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # lets "streamlit run app/demo.py" find the src package

import pandas as pd
import streamlit as st

from src.data.make_db_v2 import build_db_v2
from src.data.sql_runner import same_result
from src.evaluate import predict
from src.serve import load_slim

MODEL_PATH = ROOT / "saved_models" / "tiny_sql.pt"

st.set_page_config(page_title="Tiny text-to-SQL", layout="wide")


@st.cache_resource
def get_model():
    return load_slim(MODEL_PATH, "cpu")


def read_rows(name):
    path = ROOT / "data" / "v2" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@st.cache_data
def get_examples():
    return {
        "Wording like the training data (test set)": read_rows("test")[::80][:15],
        "New wording (paraphrase set)": read_rows("paraphrase")[::7][:15],
    }


def run_sql(conn, sql):
    """Run a SELECT and return (columns, rows, error). Anything else is refused."""
    if not sql.strip().lower().startswith("select"):
        return None, None, "Only SELECT queries are run."
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchmany(50), None
    except sqlite3.Error as e:
        return None, None, f"SQL error: {e}"


st.title("Tiny text-to-SQL")
st.caption("An 852k-parameter transformer written from scratch and trained on 8,376 generated "
           "question/SQL pairs about a small online store. It runs on your CPU.")
st.warning("On questions worded like its training data it scored 98%. On rephrased questions it scored 38%. "
           "Try one example from each set below and see the difference.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python -m src.export_model` first.")
    st.stop()

model, tok = get_model()
conn = build_db_v2()                              # built per run: cheap, and avoids SQLite thread problems
examples = get_examples()
gold = {r["question"]: r["sql"] for rows in examples.values() for r in rows}

group = st.radio("Example set", list(examples), horizontal=True)
choice = st.selectbox("Pick an example", ["(type my own)"] + [r["question"] for r in examples[group]])
question = st.text_input("Question", value="" if choice == "(type my own)" else choice,
                         key=f"q::{group}::{choice}")

if st.button("Generate SQL", type="primary") and question.strip():
    q = question.strip().lower()                  # the model was trained on lowercase text
    start = time.perf_counter()
    sql = predict(model, tok, q, "cpu")
    elapsed = time.perf_counter() - start

    st.subheader("Generated SQL")
    st.code(sql, language="sql")
    st.caption(f"{elapsed:.2f} s on CPU, {len(sql) + 1} characters generated")

    cols, rows, error = run_sql(conn, sql)
    st.subheader("Result")
    if error:
        st.error(error)
    elif not rows:
        st.info("The query ran but returned no rows.")
    else:
        st.dataframe(pd.DataFrame(rows, columns=cols))

    if q in gold:
        matches = same_result(conn, gold[q], sql)
        if matches:
            st.success("Matches the reference answer.")
        else:
            st.error("Does not match the reference answer.")
        with st.expander("Reference SQL"):
            st.code(gold[q], language="sql")

with st.expander("What is in the database?"):
    one = lambda sql: [r[0] for r in conn.execute(sql).fetchall()]
    st.write("**Cities:**", ", ".join(one("SELECT DISTINCT city FROM customers ORDER BY city")))
    st.write("**Product categories:**", ", ".join(one("SELECT DISTINCT category FROM products ORDER BY category")))
    st.write("**Some products:**", ", ".join(one("SELECT name FROM products ORDER BY id LIMIT 25")))
    st.write("**Some customers:**", ", ".join(one("SELECT name FROM customers ORDER BY id LIMIT 25")))
    st.write("**Order dates:**", "2026-01-01 to 2026-06-30")