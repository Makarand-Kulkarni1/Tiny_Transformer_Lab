import random

CITIES = [
    "pune", "mumbai", "delhi", "nagpur", "chennai", "kolkata", "nashik", "bangalore",
    "hyderabad", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur", "indore", "bhopal",
    "patna", "vadodara", "ludhiana", "agra", "varanasi", "meerut", "rajkot", "srinagar",
    "aurangabad", "dhanbad", "amritsar", "prayagraj", "ranchi", "howrah", "coimbatore",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai", "raipur", "kota",
    "guwahati", "chandigarh",
]

PRODUCTS = {
    "stationery":  ["notebook", "pen", "marker", "stapler", "folder", "eraser", "ruler",
                    "scissors", "highlighter", "diary"],
    "electronics": ["charger", "earphones", "keyboard", "mouse", "webcam", "speaker",
                    "router", "powerbank", "monitor", "tablet"],
    "kitchen":     ["kettle", "toaster", "blender", "pan", "knife", "spatula", "ladle",
                    "grater", "colander", "tongs"],
    "clothing":    ["tshirt", "jeans", "jacket", "socks", "scarf", "shirt", "shorts",
                    "sweater", "cap", "gloves"],
    "toys":        ["puzzle", "robot", "kite", "drum", "yoyo", "doll", "blocks",
                    "marbles", "spinner", "trainset"],
    "sports":      ["football", "cricketbat", "racket", "dumbbell", "skiprope", "helmet",
                    "yogamat", "shuttlecock", "bicycle", "swimcap"],
    "furniture":   ["chair", "table", "lamp", "shelf", "stool", "sofa", "desk",
                    "bookcase", "mirror", "rug"],
    "books":       ["novel", "atlas", "dictionary", "comic", "cookbook", "journal",
                    "textbook", "magazine", "biography", "poetry"],
    "beauty":      ["soap", "shampoo", "lotion", "perfume", "comb", "lipstick", "razor",
                    "conditioner", "sunscreen", "facewash"],
    "garden":      ["shovel", "hose", "planter", "rake", "seeds", "sprinkler", "trowel",
                    "wheelbarrow", "scarecrow", "fertilizer"],
    "tools":       ["hammer", "wrench", "screwdriver", "pliers", "drill", "saw",
                    "tapemeasure", "chisel", "ladder", "toolbox"],
    "pets":        ["leash", "collar", "birdcage", "fishbowl", "petbed", "chewtoy",
                    "catnip", "litterbox", "pettreat", "scratcher"],
}

CATEGORIES = list(PRODUCTS)
ALL_PRODUCT_NAMES = [name for names in PRODUCTS.values() for name in names]


def make_names(n, reserved, seed=123):
    """Generate n unique pronounceable names such as 'kavira' or 'morenu'."""
    rng = random.Random(seed)
    consonants = "bdfghjklmnprstvyz"
    vowels = "aeiou"
    names = set()
    while len(names) < n:
        syllables = rng.choice([2, 3, 3])
        name = "".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(syllables))
        if rng.random() < 0.4:
            name += rng.choice("nrslmk")
        if name not in reserved:          # never collide with a city, product or category
            names.add(name)
    return sorted(names)


FIRST_NAMES = make_names(300, set(CITIES) | set(ALL_PRODUCT_NAMES) | set(CATEGORIES))