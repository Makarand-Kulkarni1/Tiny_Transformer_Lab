import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from src.data.entities import CITIES, FIRST_NAMES, PRODUCTS

N_ORDERS = 3000


def build_db_v2(seed=42):
    """The bigger store: 300 customers, 120 products, 3000 orders. Same schema as v1."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    customers = [(i, name, rng.choice(CITIES)) for i, name in enumerate(FIRST_NAMES, start=1)]
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", customers)

    products, product_id = [], 1
    for category, names in PRODUCTS.items():
        for name in names:
            products.append((product_id, name, category, round(rng.uniform(50, 2000), 2)))
            product_id += 1
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    orders = []
    for order_id in range(1, N_ORDERS + 1):
        order_date = (date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))).isoformat()
        orders.append((order_id, rng.randint(1, len(customers)), rng.randint(1, len(products)),
                       rng.randint(1, 5), order_date))
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    conn.commit()
    return conn