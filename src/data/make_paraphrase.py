import random
from pathlib import Path

from src.data.make_dataset import save_jsonl
from src.data.make_dataset_v2 import TEMPLATES, build_pools
from src.data.make_db_v2 import build_db_v2
from src.data.sql_runner import run_query

# Two new wordings for each of the 20 templates, in the same order as TEMPLATES.
# None of them appears in the training questions.
PARAPHRASES = [
    ["how many of our customers are located in {city}?", "customer headcount for {city}"],                 # 1
    ["what do we stock in the {category} section?", "enumerate the items filed under {category}"],         # 2
    ["what would a {product} set me back?", "{product} price please"],                                    # 3
    ["how many purchases has {customer} made so far?", "what is the total number of orders {customer} has placed?"],  # 4
    ["what do we sell that is priced above {price}?", "items whose price tag is beyond {price}"],          # 5
    ["mean price for the {category} group", "give me the average of all {category} prices"],               # 6
    ["how many orders arrived later than {date}?", "orders from after {date}: how many?"],                 # 7
    ["what is the overall number of units {customer} has ordered?", "total ordered quantity for customer {customer}"],  # 8
    ["which things have been purchased by {customer}?", "show me the purchase history of {customer}"],     # 9
    ["who are the people that purchased {product}?", "buyers of {product}, please list them"],             # 10
    ["split {category} orders up by city", "per-city totals of {category} orders"],                        # 11
    ["{city} buyers: how many {category} orders?", "how many {category} purchases were made by people in {city}?"],  # 12
    ["how large is the {category} catalogue, in number of products?", "number of different products listed under {category}"],  # 13
    ["what is the most budget friendly {category} product?", "{category} item with the smallest price"],   # 14
    ["what is the top shelf {category} product by price?", "{category} item with the biggest price"],      # 15
    ["how often has {product} been purchased?", "total orders that involve {product}"],                    # 16
    ["which of our customers are based in {city}?", "customers located in {city}, list them"],             # 17
    ["how many pieces of {product} have we sold altogether?", "overall units shipped of {product}"],       # 18
    ["how many orders were received on the date {date}?", "orders count for the day {date}"],              # 19
    ["which products are priced beneath {price}?", "items that are cheaper than {price}, which are they?"],  # 20
]

PER_WORDING = 3     # how many different value fills for each wording


def main():
    assert len(PARAPHRASES) == len(TEMPLATES), "need one pair of wordings per template"
    training_questions = {q for t in TEMPLATES for q in t["questions"]}
    assert not any(w in training_questions for pair in PARAPHRASES for w in pair), "a paraphrase is also a training wording"

    rng = random.Random(1)
    pool = build_pools(random.Random(0))["test"]     # the same held-out test values as make_dataset_v2
    conn = build_db_v2()

    pairs = set()
    for template, wordings in zip(TEMPLATES, PARAPHRASES):
        for wording in wordings:
            made, tries = 0, 0
            while made < PER_WORDING and tries < 500:
                tries += 1
                slots = {key: rng.choice(values) for key, values in pool.items()}
                question = wording.format(**slots)
                sql = template["sql"].format(**slots)
                result = run_query(conn, sql)
                if not result or result in ([(0,)], [(None,)]):   # skip trivial answers
                    continue
                if (question, sql) in pairs:
                    continue
                pairs.add((question, sql))
                made += 1

    out = Path(__file__).resolve().parents[2] / "data" / "v2" / "paraphrase.jsonl"
    save_jsonl(sorted(pairs), out)
    print(len(pairs), "paraphrase examples written to", out)


if __name__ == "__main__":
    main()