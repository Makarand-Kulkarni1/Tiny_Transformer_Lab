import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from src.data.make_db import build_db, FIRST_NAMES, CITIES, PRODUCTS
from src.data.sql_runner import run_query

ALL_PRODUCT_NAMES = [name for names in PRODUCTS.values() for name in names]


def random_slots(rng):
    """Make one random value for every possible blank. Templates use whichever they need."""
    return {
        "city": rng.choice(CITIES),
        "category": rng.choice(list(PRODUCTS)),
        "product": rng.choice(ALL_PRODUCT_NAMES),
        "customer": rng.choice(FIRST_NAMES),
        "price": rng.randrange(100, 2000, 50),
        "date": (date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))).isoformat(),
        "qty": rng.randint(1, 4),
    }


TEMPLATES = [
    {   # 1: count customers in a city
        "questions": [
            "how many customers live in {city}?",
            "count the customers from {city}",
            "what is the number of customers in {city}?",
        ],
        "sql": "SELECT COUNT(*) FROM customers WHERE city = '{city}'",
    },
    {   # 2: list products in a category
        "questions": [
            "list all products in the {category} category",
            "which products are in {category}?",
            "show the names of {category} products",
        ],
        "sql": "SELECT name FROM products WHERE category = '{category}'",
    },
    {   # 3: price of one product
        "questions": [
            "what is the price of {product}?",
            "how much does {product} cost?",
            "give me the price of the {product}",
        ],
        "sql": "SELECT price FROM products WHERE name = '{product}'",
    },
    {   # 4: orders placed by one customer (needs a JOIN)
        "questions": [
            "how many orders did {customer} place?",
            "count the orders made by {customer}",
            "what is the number of orders from {customer}?",
        ],
        "sql": "SELECT COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.name = '{customer}'",
    },
    {   # 5: products above a price
        "questions": [
            "which products cost more than {price}?",
            "list the products priced above {price}",
            "show the names of products with a price greater than {price}",
        ],
        "sql": "SELECT name FROM products WHERE price > {price}",
    },
    {   # 6: average price in a category
        "questions": [
            "what is the average price of {category} products?",
            "average price in the {category} category",
            "how much do {category} products cost on average?",
        ],
        "sql": "SELECT AVG(price) FROM products WHERE category = '{category}'",
    },
    {   # 7: orders after a date
        "questions": [
            "how many orders were placed after {date}?",
            "count the orders made after {date}",
            "what is the number of orders since {date}?",
        ],
        "sql": "SELECT COUNT(*) FROM orders WHERE order_date > '{date}'",
    },
    {   # 8: total quantity ordered by a customer
        "questions": [
            "what is the total quantity ordered by {customer}?",
            "how many items in total did {customer} order?",
            "sum the quantities ordered by {customer}",
        ],
        "sql": "SELECT SUM(quantity) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.name = '{customer}'",
    },
    {   # 9: products ordered by a customer
        "questions": [
            "which products did {customer} order?",
            "list the products ordered by {customer}",
            "what has {customer} bought?",
        ],
        "sql": "SELECT DISTINCT products.name FROM orders JOIN customers ON orders.customer_id = customers.id JOIN products ON orders.product_id = products.id WHERE customers.name = '{customer}'",
    },
    {   # 10: customers who ordered a product
        "questions": [
            "which customers ordered {product}?",
            "list the customers who bought {product}",
            "who has ordered the {product}?",
        ],
        "sql": "SELECT DISTINCT customers.name FROM orders JOIN customers ON orders.customer_id = customers.id JOIN products ON orders.product_id = products.id WHERE products.name = '{product}'",
    },
    {   # 11: orders per city for one category (GROUP BY)
        "questions": [
            "how many {category} orders were placed in each city?",
            "number of {category} orders per city",
            "count the {category} orders for every city",
        ],
        "sql": "SELECT customers.city, COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id JOIN products ON orders.product_id = products.id WHERE products.category = '{category}' GROUP BY customers.city",
    },
    {   # 12: orders of a category from customers in a city
        "questions": [
            "how many {category} orders came from {city}?",
            "count the {category} orders placed by customers in {city}",
            "what is the number of {category} orders from {city} customers?",
        ],
        "sql": "SELECT COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id JOIN products ON orders.product_id = products.id WHERE products.category = '{category}' AND customers.city = '{city}'",
    },
]


def generate_pairs(conn, rng, attempts=40000):
    pairs = set()          # a set removes duplicates automatically
    skipped = 0
    for _ in range(attempts):
        template = rng.choice(TEMPLATES)
        slots = random_slots(rng)
        question = rng.choice(template["questions"]).format(**slots)
        sql = template["sql"].format(**slots)

        result = run_query(conn, sql)
        if not result:     # None (error) or [] (no rows): useless for scoring
            skipped += 1
            continue
        pairs.add((question, sql))

    print(f"unique pairs: {len(pairs)}, skipped: {skipped}")
    return sorted(pairs)   # sorting makes the order the same on every run


def skeleton(sql):
    """Mask quoted values and numbers, so every SQL from one template looks identical."""
    sql = re.sub(r"'[^']*'", "?", sql)
    return re.sub(r"\b\d+(\.\d+)?\b", "?", sql)


def split_by_sql(pairs, rng):
    """Split each template separately, so every template appears in train, val and test.
    Each SQL still lives in exactly one split (no leakage)."""
    groups = {}                                    # skeleton -> set of unique SQL strings
    for _, sql in pairs:
        groups.setdefault(skeleton(sql), set()).add(sql)

    train_sql, val_sql = set(), set()
    for key in sorted(groups):                     # sorted keeps runs reproducible
        sqls = sorted(groups[key])
        rng.shuffle(sqls)
        n_hold = max(1, round(0.1 * len(sqls)))    # at least 1 SQL each for val and test
        val_sql.update(sqls[:n_hold])              # first slice -> validation
        train_sql.update(sqls[2 * n_hold:])        # the slice after it falls through to test

    splits = {"train": [], "val": [], "test": []}
    for question, sql in pairs:
        if sql in train_sql:
            splits["train"].append((question, sql))
        elif sql in val_sql:
            splits["val"].append((question, sql))
        else:
            splits["test"].append((question, sql))
    return splits


def save_jsonl(rows, path):
    """Write one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # create data/raw if it is missing
    with open(path, "w", encoding="utf-8") as f:
        for question, sql in rows:
            f.write(json.dumps({"question": question, "sql": sql}) + "\n")


if __name__ == "__main__":
    conn = build_db()
    rng = random.Random(0)

    pairs = generate_pairs(conn, rng)
    splits = split_by_sql(pairs, rng)

    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    for name, rows in splits.items():
        save_jsonl(rows, out_dir / f"{name}.jsonl")
        print(name, len(rows))