import json
import random

# CONFIGURACIÓN
OUTPUT_FILE = "dataset_v2.jsonl"
NUM_EXAMPLES = 500  # Generamos más ejemplos para cubrir todas las combinaciones

# SYSTEM PROMPT (Estricto en Inglés, con tus Tools reales)
SYSTEM_PROMPT = """You are a JSON-only API Assistant for a fashion retailer.
You define the user intent by selecting the correct tool.

RULES:
1. Output ONLY valid JSON.
2. Do not speak or explain. Only return the JSON object.

TOOLS:
- search_products(query: string) -> Search for clothes by keywords, brand, or style.
- refine_products(color: string, size: string, max_price: number, brand: string) -> Filter current results.
- get_product_by_sku(sku: string) -> Get details for a specific 9-digit SKU.
- business_info(topic: string) -> Info about shipping, returns, payment.
- chat_response(message: string) -> Greetings and general chat.
"""

# ==========================================
# VOCABULARIO EXTRAÍDO DE TU CSV
# ==========================================

brands = ["New Look", "Stradivarius", "ASOS DESIGN", "JDY", "Nike Running", "Bershka", "Barneys Originals", "Topshop"]

# 1. Los términos complejos de tu CSV (para que sea preciso)
csv_items = ["trench coat", "faux leather biker jacket", "wool coat", "longline trench", "hooded jacket", "racer jacket"]

# 2. Términos genéricos que usan los humanos (para que sea flexible)
generic_items = ["coat", "jacket", "shirt", "pants", "dress", "outfit", "clothes", "something", "top"]

# Unimos ambas listas para el entrenamiento
items = csv_items + generic_items

colors = ["Camel", "Black", "Stone", "Khaki", "Grey", "Pink", "Neutral", "White"]

sizes = ["UK 4", "UK 6", "UK 8", "UK 10", "UK 12", "XS", "S", "M", "L", "XL"]

# SKUs reales de tu CSV (para que aprenda el formato de 9 dígitos)
skus = ["126704571", "203490700", "203439012", "201104221", "203438897", "123650194", "125806824", "121963507", "204166264"]

adjectives = ["oversized", "double breasted", "longline", "petite", "water-repellent", "biker"]

# ==========================================
# GENERADORES
# ==========================================

def gen_greeting():
    prompts = [
        "Hi", "Hello", "Good morning", "Hey", "Are you there?", 
        "I need help finding a coat", "Start", "Menu"
    ]
    inp = random.choice(prompts)
    out = {"name": "chat_response", "arguments": {"message": "Hello! I can help you find jackets, coats, and more."}}
    return inp, out

def gen_search():
    # Patrones de búsqueda basados en cómo busca la gente ropa real
    brand = random.choice(brands)
    item = random.choice(items)
    color = random.choice(colors)
    adj = random.choice(adjectives)
    
    generic_templates = [
        (f"I need some {item}", item),
        (f"Do you have {color} clothes?", color),
        (f"Show me something {random.choice(['cheap', 'nice', 'popular'])}", ""), # Query vacía o palabra clave
        (f"I want to buy a {item}", item),
        (f"Looking for {item}s", item)
    ]
    
    specific_templates = [
        (f"I want a {brand} {item}", f"{brand} {item}"),
        (f"Show me {color} {item}s", f"{color} {item}"),
        (f"Search for {item} in {color}", f"{item} {color}"),
        (f"Do you have {adj} jackets?", f"{adj} jacket"),
        (f"Looking for {brand}", brand),
        (f"I need a {item} for winter", item),
        (f"Show me the {brand} {adj} {item}", f"{brand} {adj} {item}")
    ]
    
    # 50% de probabilidad de ser muy específico (tipo CSV) vs muy genérico
    if random.random() < 0.5:
        inp, query = random.choice(specific_templates)
    else:
        inp, query = random.choice(generic_templates)
        
    out = {"name": "search_products", "arguments": {"query": query}}
    return inp, out

def gen_refine():
    # Simulamos filtros específicos de tu inventario
    case = random.choice(["color", "size", "price", "brand", "mixed"])
    
    if case == "color":
        c = random.choice(colors)
        inp = f"Show me in {c}"
        args = {"color": c}
    elif case == "size":
        s = random.choice(sizes)
        inp = f"Do you have size {s}?"
        args = {"size": s}
    elif case == "price":
        p = random.choice([50, 60, 100, 45])
        inp = f"My budget is {p}"
        args = {"max_price": p}
    elif case == "brand":
        b = random.choice(brands)
        inp = f"Only from {b}"
        args = {"brand": b}
    else: # Mixed
        c = random.choice(colors)
        s = random.choice(sizes)
        inp = f"{c} and size {s} please"
        args = {"color": c, "size": s}

    out = {"name": "refine_products", "arguments": args}
    return inp, out

def gen_sku():
    sku = random.choice(skus)
    templates = [
        f"Show me product {sku}",
        f"I want to see the item with code {sku}",
        f"Search SKU {sku}",
        f"{sku}",
        f"Open details for {sku}"
    ]
    inp = random.choice(templates)
    out = {"name": "get_product_by_sku", "arguments": {"sku": sku}}
    return inp, out

def gen_business():
    # Preguntas típicas de e-commerce
    shipping_qs = ["Do you ship to UK?", "How long is delivery?", "Shipping cost?"]
    return_qs = ["What is the return policy?", "Can I return if it doesn't fit?", "How to return?"]
    
    topic = random.choice(["shipping", "returns"])
    if topic == "shipping":
        inp = random.choice(shipping_qs)
    else:
        inp = random.choice(return_qs)
        
    out = {"name": "business_info", "arguments": {"topic": topic}}
    return inp, out

# ==========================================
# CONSTRUCCIÓN DEL DATASET
# ==========================================

data = []

for _ in range(NUM_EXAMPLES):
    r = random.random()
    
    if r < 0.10:
        inp, out = gen_greeting()
    elif r < 0.40:
        inp, out = gen_search()    # Mayor peso a búsqueda
    elif r < 0.65:
        inp, out = gen_refine()    # Filtros
    elif r < 0.85:
        inp, out = gen_sku()       # SKUs (Importante por tu CSV)
    else:
        inp, out = gen_business()

    entry = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": json.dumps(out)}
        ]
    }
    data.append(entry)

# Guardar
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Dataset generado con éxito: {OUTPUT_FILE} ({NUM_EXAMPLES} ejemplos)")
print("   - Vocabulario adaptado a: New Look, Stradivarius, Nike, etc.")
print("   - Formatos de SKU reales.")