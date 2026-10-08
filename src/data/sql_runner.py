import sqlite3


def run_query(conn, sql):
    """Run a SELECT query and return its rows sorted, or None if anything is wrong."""
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):      # safety: never run anything that isn't a read
        return None
    try:
        rows = conn.execute(sql).fetchall()   # a list of tuples, one per result row
    except sqlite3.Error:                     # broken SQL must not crash our evaluation
        return None
    return sorted(rows, key=repr)             # sort so row order doesn't matter


def same_result(conn, gold_sql, predicted_sql):
    """True only if both queries run and give the same rows."""
    gold = run_query(conn, gold_sql)
    predicted = run_query(conn, predicted_sql)
    if gold is None or predicted is None:     # a failed query is never "correct"
        return False
    return gold == predicted