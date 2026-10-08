import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

FIRST_NAMES = [
    "asha", "rohan", "meera", "kabir", "sneha", "vikram", "anika", "arjun",
    "diya", "ishaan", "kavya", "neel", "priya", "rahul", "sana", "tara",
    "uday", "veda", "yash", "zoya", "aarav", "bhavna", "chetan", "deepa",
    "esha", "farhan", "gauri", "harsh", "isha", "jay", "kiran", "leela",
    "mohan", "nisha", "omkar", "pooja", "qadir", "riya", "sameer", "tanvi",
]
CITIES = ["pune", "mumbai", "delhi", "nagpur", "chennai", "kolkata"]
PRODUCTS = {
    "stationery":  ["notebook", "pen", "marker", "stapler", "folder"],
    "electronics": ["charger", "earphones", "keyboard", "mouse", "webcam"],
    "kitchen":     ["kettle", "toaster", "blender", "pan", "knife"],
    "clothing":    ["tshirt", "jeans", "jacket", "socks", "scarf"],
    "toys":        ["puzzle", "robot", "kite", "drum", "yoyo"],
}
def build_db(seed=42):
    rng = random.Random(seed)              # our own dice, always rolling the same sequence
    conn = sqlite3.connect(":memory:")     # a database that lives only in RAM

    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())   # create the three empty tables

    # ---- customers ----
    customers = []
    for i, name in enumerate(FIRST_NAMES, start=1):
        city = rng.choice(CITIES)
        customers.append((i, name, city))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", customers)

    # ---- products ----
    products = []
    product_id = 1
    for category, names in PRODUCTS.items():
        for name in names:
            price = round(rng.uniform(50, 2000), 2)
            products.append((product_id, name, category, price))
            product_id += 1
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    # ---- orders ----
    orders = []
    for order_id in range(1, 401):
        customer_id = rng.randint(1, 40)
        product_id = rng.randint(1, 25)
        quantity = rng.randint(1, 5)
        order_date = (date(2026, 1, 1) + timedelta(days=rng.randint(0, 180))).isoformat()
        orders.append((order_id, customer_id, product_id, quantity, order_date))
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    conn.commit()
    return conn


if __name__ == "__main__":
    conn = build_db()
    print(conn.execute("SELECT COUNT(*) FROM customers").fetchone())
    print(conn.execute("SELECT COUNT(*) FROM products").fetchone())
    print(conn.execute("SELECT COUNT(*) FROM orders").fetchone())