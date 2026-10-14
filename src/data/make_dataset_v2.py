import random
from datetime import date, timedelta
from pathlib import Path

from src.data.entities import CATEGORIES, CITIES, FIRST_NAMES, ALL_PRODUCT_NAMES
from src.data.make_dataset import skeleton, save_jsonl
from src.data.make_db_v2 import build_db_v2
from src.data.sql_runner import run_query

DATES = [(date(2026, 1, 1) + timedelta(days=d)).isoformat() for d in range(181)]
PRICES = list(range(100, 2000, 50))       # 38 values

JOIN_C = "JOIN customers ON orders.customer_id = customers.id"
JOIN_P = "JOIN products ON orders.product_id = products.id"

TEMPLATES = [
    {   # 1: count customers in a city
        "questions": [
            "how many customers live in {city}?",
            "count the customers from {city}",
            "what is the number of customers in {city}?",
            "number of customers based in {city}",
            "how many people from {city} are customers?",
            "tell me the customer count for {city}",
            "{city} has how many customers?",
            "count customers whose city is {city}",
        ],
        "sql": "SELECT COUNT(*) FROM customers WHERE city = '{city}'",
    },
    {   # 2: list products in a category
        "questions": [
            "list all products in the {category} category",
            "which products are in {category}?",
            "show the names of {category} products",
            "what products do we have under {category}?",
            "give me every product in {category}",
            "name all the {category} items",
            "display the products of the {category} category",
            "i want to see all {category} products",
        ],
        "sql": "SELECT name FROM products WHERE category = '{category}'",
    },
    {   # 3: price of one product
        "questions": [
            "what is the price of {product}?",
            "how much does {product} cost?",
            "give me the price of the {product}",
            "price of {product}?",
            "what does the {product} sell for?",
            "tell me how much a {product} costs",
            "show the price for {product}",
            "how expensive is the {product}?",
        ],
        "sql": "SELECT price FROM products WHERE name = '{product}'",
    },
    {   # 4: count orders of a customer
        "questions": [
            "how many orders did {customer} place?",
            "count the orders made by {customer}",
            "what is the number of orders from {customer}?",
            "how many times has {customer} ordered?",
            "number of orders placed by {customer}",
            "{customer} made how many orders?",
            "tell me the order count for {customer}",
            "count all orders belonging to {customer}",
        ],
        "sql": f"SELECT COUNT(*) FROM orders {JOIN_C} WHERE customers.name = '{{customer}}'",
    },
    {   # 5: products above a price
        "questions": [
            "which products cost more than {price}?",
            "list the products priced above {price}",
            "show the names of products with a price greater than {price}",
            "what items are more expensive than {price}?",
            "give me products that cost over {price}",
            "find all products priced higher than {price}",
            "name the products whose price exceeds {price}",
            "which items sell for more than {price}?",
        ],
        "sql": "SELECT name FROM products WHERE price > {price}",
    },
    {   # 6: average price in a category
        "questions": [
            "what is the average price of {category} products?",
            "average price in the {category} category",
            "how much do {category} products cost on average?",
            "tell me the mean price of {category} items",
            "what is the typical price in {category}?",
            "compute the average cost of {category} products",
            "average {category} product price?",
            "on average, what do {category} products cost?",
        ],
        "sql": "SELECT AVG(price) FROM products WHERE category = '{category}'",
    },
    {   # 7: orders after a date
        "questions": [
            "how many orders were placed after {date}?",
            "count the orders made after {date}",
            "what is the number of orders since {date}?",
            "number of orders later than {date}",
            "how many orders came in after {date}?",
            "count all orders dated after {date}",
            "after {date}, how many orders were there?",
            "tell me how many orders were made following {date}",
        ],
        "sql": "SELECT COUNT(*) FROM orders WHERE order_date > '{date}'",
    },
    {   # 8: total quantity ordered by a customer
        "questions": [
            "what is the total quantity ordered by {customer}?",
            "how many items in total did {customer} order?",
            "sum the quantities ordered by {customer}",
            "total number of units bought by {customer}",
            "how many units has {customer} purchased in all?",
            "add up everything {customer} ordered",
            "what quantity did {customer} order altogether?",
            "tell me the total items ordered by {customer}",
        ],
        "sql": f"SELECT SUM(quantity) FROM orders {JOIN_C} WHERE customers.name = '{{customer}}'",
    },
    {   # 9: products ordered by a customer
        "questions": [
            "which products did {customer} order?",
            "list the products ordered by {customer}",
            "what has {customer} bought?",
            "show every product {customer} purchased",
            "what did {customer} order?",
            "give me the items bought by {customer}",
            "name the products {customer} has ordered",
            "which items has {customer} purchased?",
        ],
        "sql": f"SELECT DISTINCT products.name FROM orders {JOIN_C} {JOIN_P} WHERE customers.name = '{{customer}}'",
    },
    {   # 10: customers who ordered a product
        "questions": [
            "which customers ordered {product}?",
            "list the customers who bought {product}",
            "who has ordered the {product}?",
            "show everyone who purchased {product}",
            "which people bought a {product}?",
            "give me the names of customers that ordered {product}",
            "who bought the {product}?",
            "name all buyers of {product}",
        ],
        "sql": f"SELECT DISTINCT customers.name FROM orders {JOIN_C} {JOIN_P} WHERE products.name = '{{product}}'",
    },
    {   # 11: orders per city for one category
        "questions": [
            "how many {category} orders were placed in each city?",
            "number of {category} orders per city",
            "count the {category} orders for every city",
            "for each city, how many {category} orders are there?",
            "give me the {category} order count by city",
            "break down {category} orders by city",
            "city wise count of {category} orders",
            "how many orders of {category} items did each city place?",
        ],
        "sql": f"SELECT customers.city, COUNT(*) FROM orders {JOIN_C} {JOIN_P} WHERE products.category = '{{category}}' GROUP BY customers.city",
    },
    {   # 12: orders of a category from a city
        "questions": [
            "how many {category} orders came from {city}?",
            "count the {category} orders placed by customers in {city}",
            "what is the number of {category} orders from {city} customers?",
            "number of {category} orders made in {city}",
            "how many times did {city} customers order {category} items?",
            "count {category} orders from {city}",
            "{city} customers placed how many {category} orders?",
            "tell me the {category} order count for {city}",
        ],
        "sql": f"SELECT COUNT(*) FROM orders {JOIN_C} {JOIN_P} WHERE products.category = '{{category}}' AND customers.city = '{{city}}'",
    },
    {   # 13: count products in a category
        "questions": [
            "how many products are in the {category} category?",
            "count the {category} products",
            "what is the number of products under {category}?",
            "number of items in {category}",
            "how many {category} items do we sell?",
            "tell me the product count for {category}",
            "{category} has how many products?",
            "count all products belonging to {category}",
        ],
        "sql": "SELECT COUNT(*) FROM products WHERE category = '{category}'",
    },
    {   # 14: cheapest product in a category
        "questions": [
            "which {category} product is the cheapest?",
            "what is the cheapest product in {category}?",
            "name the lowest priced {category} item",
            "find the least expensive {category} product",
            "cheapest item in the {category} category?",
            "which product in {category} costs the least?",
            "show the {category} product with the lowest price",
            "tell me the most affordable {category} product",
        ],
        "sql": "SELECT name FROM products WHERE category = '{category}' ORDER BY price ASC LIMIT 1",
    },
    {   # 15: most expensive product in a category
        "questions": [
            "which {category} product is the most expensive?",
            "what is the priciest product in {category}?",
            "name the highest priced {category} item",
            "find the most costly {category} product",
            "most expensive item in the {category} category?",
            "which product in {category} costs the most?",
            "show the {category} product with the highest price",
            "tell me the top priced {category} product",
        ],
        "sql": "SELECT name FROM products WHERE category = '{category}' ORDER BY price DESC LIMIT 1",
    },
    {   # 16: count orders of a product
        "questions": [
            "how many orders were placed for {product}?",
            "count the orders of {product}",
            "what is the number of orders for the {product}?",
            "how many times was {product} ordered?",
            "number of orders containing {product}",
            "{product} was ordered how many times?",
            "tell me the order count for {product}",
            "count all orders that include {product}",
        ],
        "sql": f"SELECT COUNT(*) FROM orders {JOIN_P} WHERE products.name = '{{product}}'",
    },
    {   # 17: list customers in a city
        "questions": [
            "which customers live in {city}?",
            "list the customers from {city}",
            "show the names of customers in {city}",
            "who lives in {city}?",
            "give me every customer based in {city}",
            "name all customers whose city is {city}",
            "display the customers of {city}",
            "i want the customer names for {city}",
        ],
        "sql": "SELECT name FROM customers WHERE city = '{city}'",
    },
    {   # 18: total quantity of a product
        "questions": [
            "what is the total quantity ordered of {product}?",
            "how many units of {product} were ordered in total?",
            "sum the quantities ordered for {product}",
            "total units sold of {product}",
            "how many {product} items have been bought altogether?",
            "total quantity sold for {product}",
            "what quantity of {product} was ordered overall?",
            "tell me the total units of {product} ordered",
        ],
        "sql": f"SELECT SUM(quantity) FROM orders {JOIN_P} WHERE products.name = '{{product}}'",
    },
    {   # 19: orders on a date
        "questions": [
            "how many orders were placed on {date}?",
            "count the orders made on {date}",
            "what is the number of orders on {date}?",
            "number of orders dated {date}",
            "how many orders came in on {date}?",
            "count all orders from {date}",
            "on {date}, how many orders were there?",
            "tell me the order count for {date}",
        ],
        "sql": "SELECT COUNT(*) FROM orders WHERE order_date = '{date}'",
    },
    {   # 20: products below a price
        "questions": [
            "which products cost less than {price}?",
            "list the products priced below {price}",
            "show the names of products with a price lower than {price}",
            "what items are cheaper than {price}?",
            "give me products that cost under {price}",
            "find all products priced less than {price}",
            "name the products whose price is under {price}",
            "which items sell for less than {price}?",
        ],
        "sql": "SELECT name FROM products WHERE price < {price}",
    },
]


def split_pool(items, rng, min_hold=2):
    """Divide a list of values into disjoint train / val / test pools (about 80/10/10)."""
    items = sorted(items)
    rng.shuffle(items)
    n_hold = max(min_hold, round(0.1 * len(items)))
    return {"val": items[:n_hold], "test": items[n_hold:2 * n_hold], "train": items[2 * n_hold:]}


def build_pools(rng):
    """Every value (a city, a name, a date...) belongs to exactly one split."""
    raw = {"city": CITIES, "category": CATEGORIES, "product": ALL_PRODUCT_NAMES,
           "customer": FIRST_NAMES, "price": PRICES, "date": DATES}
    split = {key: split_pool(values, rng) for key, values in raw.items()}
    return {s: {key: split[key][s] for key in raw} for s in ("train", "val", "test")}


def generate_split(conn, rng, pool, cap):
    """Up to `cap` unique pairs per template, using only this split's values."""
    pairs, usable = set(), {}                  # usable caches whether each SQL is worth keeping
    for template in TEMPLATES:
        mine, stale = set(), 0
        while len(mine) < cap and stale < 3000:
            slots = {key: rng.choice(values) for key, values in pool.items()}
            question = rng.choice(template["questions"]).format(**slots)
            sql = template["sql"].format(**slots)
            if sql not in usable:
                result = run_query(conn, sql)
                usable[sql] = bool(result) and result not in ([(0,)], [(None,)])
            if not usable[sql]:                # error, empty, 0 or NULL: too easy to guess
                stale += 1
                continue
            before = len(mine)
            mine.add((question, sql))
            stale = 0 if len(mine) > before else stale + 1
        pairs |= mine
    return sorted(pairs)


if __name__ == "__main__":
    rng = random.Random(0)
    conn = build_db_v2()
    pools = build_pools(rng)
    out_dir = Path(__file__).resolve().parents[2] / "data" / "v2"

    splits = {}
    for split, cap in [("train", 700), ("val", 100), ("test", 100)]:
        splits[split] = generate_split(conn, rng, pools[split], cap)
        save_jsonl(splits[split], out_dir / f"{split}.jsonl")

    sql_sets = {s: {sql for _, sql in pairs} for s, pairs in splits.items()}
    assert not sql_sets["train"] & sql_sets["val"], "leakage: train/val share a SQL"
    assert not sql_sets["train"] & sql_sets["test"], "leakage: train/test share a SQL"
    assert not sql_sets["val"] & sql_sets["test"], "leakage: val/test share a SQL"

    for s, pairs in splits.items():
        print(f"{s:5s} {len(pairs):6d} pairs | templates present: {len({skeleton(sql) for _, sql in pairs})}")
    print("longest example:", max(len(q) + len(sql) + 2 for p in splits.values() for q, sql in p), "characters")
    print("no leakage between splits")