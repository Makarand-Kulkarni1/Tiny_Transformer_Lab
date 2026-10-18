import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

from src.data.make_dataset import skeleton
from src.data.sql_runner import run_query, same_result
from src.evaluate import DB_BUILDERS, per_template, short, summarize
from src.train import ROOT, load_jsonl


class RateLimitStop(Exception):
    """Raised when the API asks us to wait too long; we save what we have and stop."""


def read_key(env_name):
    """The key comes from an environment variable, or from a line KEY=value in .env (never committed)."""
    key = os.environ.get(env_name)
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{env_name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def pick_shots(train_rows, n):
    """One training example from each of n different templates, always the same ones."""
    first = {}
    for r in train_rows:
        first.setdefault(skeleton(r["sql"]), r)
    keys = sorted(first)
    step = max(1, len(keys) // n)
    return [first[k] for k in keys[::step][:n]]

def build_prefix(train_rows, shots):
    schema = (Path(__file__).parent / "data" / "schema.sql").read_text()
    system = ("You convert questions into SQLite queries. The database schema is:\n"
              f"{schema}\n"
              "Every text value stored in the database is lowercase, so write values in lowercase "
              "exactly as they appear in the question. "
              "Reply with exactly one SQLite SELECT query and nothing else: no explanation and no markdown.")
    messages = [{"role": "system", "content": system}]
    for r in pick_shots(train_rows, shots):
        messages.append({"role": "user", "content": r["question"]})
        messages.append({"role": "assistant", "content": r["sql"]})
    return messages


def clean(text):
    """Strip markdown fences and the trailing semicolon, and collapse whitespace."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    return " ".join(text.split()).rstrip(";").strip()


def ask(session, args, key, messages):
    url = f"{args.base_url}/chat/completions"
    payload = {"model": args.model, "messages": messages, "temperature": 0,
               "max_completion_tokens": args.max_tokens}
    if args.reasoning_effort and args.reasoning_effort != "off":
        payload["reasoning_effort"] = args.reasoning_effort
    for attempt in range(6):
        start = time.time()
        try:
            resp = session.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=120)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = 5 * 2 ** attempt
            print(f"  connection problem ({type(e).__name__}), retrying in {wait}s")
            session.close()                      # drop the broken connection; the next call opens a new one
            time.sleep(wait)
            continue
        seconds = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"].get("content") or ""
            return content, seconds, data.get("usage", {})
        if resp.status_code in (429, 500, 502, 503, 504):
            try:
                wait = float(resp.headers.get("retry-after", 5 * 2 ** attempt))
            except ValueError:
                wait = 5 * 2 ** attempt
            if wait > 180:
                raise RateLimitStop(f"the API asked us to wait {wait / 60:.0f} minutes: {resp.text[:200]}")
            print(f"  status {resp.status_code}, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        raise SystemExit(f"API error {resp.status_code}: {resp.text[:300]}")
    raise RateLimitStop("too many retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="api")
    parser.add_argument("--data", default="v2", choices=["v1", "v2"])
    parser.add_argument("--split", default="paraphrase", choices=["val", "test", "paraphrase"])
    parser.add_argument("--limit", type=int, default=0, help="evaluate a random subset (same sampling as evaluate.py)")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max_tokens", type=int, default=1500, help="output budget; reasoning models spend part of it thinking")
    parser.add_argument("--reasoning_effort", default="low", help="low/medium/high for gpt-oss; 'off' leaves it out")
    parser.add_argument("--base_url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--key_env", default="GROQ_API_KEY")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=4.0, help="seconds between requests (rate limits)")
    args = parser.parse_args()

    key = read_key(args.key_env)
    if not key:
        raise SystemExit(f"Set the {args.key_env} environment variable first (see the instructions).")

    prefix = build_prefix(load_jsonl("train", args.data), args.shots)
    rows = load_jsonl(args.split, args.data)
    if args.limit and args.limit < len(rows):
        rows = random.Random(0).sample(rows, args.limit)
    conn = DB_BUILDERS[args.data]()

    (ROOT / "results").mkdir(exist_ok=True)
    cache_path = ROOT / "results" / f"{args.run_name}_{args.split}_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["question"]] = rec
    print(f"model {args.model} | {len(rows)} questions | {len(cache)} already answered")

    session, stopped, done_now = requests.Session(), None, 0
    for row in rows:
        if row["question"] in cache:
            continue
        try:
            text, seconds, usage = ask(session, args, key, prefix + [{"role": "user", "content": row["question"]}])
        except RateLimitStop as e:
            stopped = str(e)
            break
        rec = {"question": row["question"], "predicted": clean(text), "seconds": seconds,
               "n_tokens": max(1, usage.get("completion_tokens", 0)),
               "prompt_tokens": usage.get("prompt_tokens", 0)}
        cache[row["question"]] = rec
        if not rec["predicted"]:
            print("  warning: empty answer (try a larger --max_tokens)")
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done_now += 1
        if done_now % 20 == 0:
            print(f"  {len(cache)} answered")
        time.sleep(args.sleep)

    results = []
    for row in rows:
        rec = cache.get(row["question"])
        if rec is None:
            continue
        pred = rec["predicted"]
        results.append({
            "question": row["question"], "gold": row["sql"], "predicted": pred,
            "exact": pred == row["sql"], "valid": run_query(conn, pred) is not None,
            "exec_correct": same_result(conn, row["sql"], pred),
            "seconds": rec["seconds"], "n_tokens": rec["n_tokens"],
        })
    if not results:
        raise SystemExit("No answers yet. " + (stopped or ""))

    complete = len(results) == len(rows)
    summary = summarize(results)
    templates = per_template(results)
    summary.update({
        "macro_exec_accuracy": sum(s["exec_accuracy"] for s in templates.values()) / len(templates),
        "run_name": args.run_name, "data": args.data, "split": args.split, "model": args.model,
        "shots": args.shots, "complete": complete,
        "avg_prompt_tokens": sum(cache[r["question"]]["prompt_tokens"] for r in rows if r["question"] in cache) / len(results),
    })

    print(f"\n=== overall ({len(results)} of {len(rows)} questions{'' if complete else ' - INCOMPLETE'}) ===")
    print(f"execution accuracy:       {summary['exec_accuracy']:.1%}")
    print(f"macro (per-template avg): {summary['macro_exec_accuracy']:.1%}")
    print(f"exact match:              {summary['exact_match']:.1%}")
    print(f"valid SQL:                {summary['valid_sql']:.1%}")
    print(f"latency:                  {summary['avg_seconds']:.2f} s per question (network + model)")
    print(f"prompt size:              {summary['avg_prompt_tokens']:.0f} tokens per question")

    print("\n=== per template ===")
    print(f"{'template':95s} {'n':>4s} {'exec':>7s}")
    for k, s in templates.items():
        print(f"{short(k)[:95]:95s} {s['n']:4d} {s['exec_accuracy']:7.1%}")

    failures = [r for r in results if not r["exec_correct"]]
    print(f"\n=== failures: {len(failures)} (showing up to 5) ===")
    for r in failures[:5]:
        print("Q:    ", r["question"])
        print("gold: ", r["gold"])
        print("pred: ", r["predicted"])
        print()

    out = ROOT / "results" / f"{args.run_name}_{args.split}_eval.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_template": templates, "examples": results}, f, indent=2)
    print("saved", out)
    if stopped:
        print("\nStopped early:", stopped, "\nRun the same command again later to continue where it stopped.")


if __name__ == "__main__":
    main()