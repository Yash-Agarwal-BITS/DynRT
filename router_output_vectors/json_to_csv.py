import os
import json
import csv

# --- CONFIGURE THESE ---
BASE_DIR   = 'MMSD2.0 (temp = 0.1)'
INPUT_JSON = os.path.join(BASE_DIR, 'router_output_vectors_tau(0.1).json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'router_output_vectors_tau(0.1)_sorted.csv')
# ------------------------

# Define desired order for types
type_order = {
    'non sarcastic': 0,
    'object': 1,
    'sentimental': 2,
    'situational': 3
}

# 1. Load JSON
data = []
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

# 2. Normalize types and prepare flattened rows
dataset = []
for rec in data:
    lbl = rec['label']
    # Determine canonical type
    if lbl == 0:
        t = 'non sarcastic'
    else:
        raw_t = rec.get('type', '').strip().lower()
        if 'object' in raw_t:
            t = 'object'
        elif 'sentiment' in raw_t:
            t = 'sentimental'
        elif 'situational' in raw_t:
            t = 'situational'
        else:
            t = raw_t  # fallback

    # Build flat row dict
    row = {
        'image_id': rec['image_id'],
        'label': lbl,
        'type': t
    }

    rv = rec['router_vectors']
    # Layer 0 (scalar)
    row['router_vector_0'] = rv[0]

    # Layers 1-3 (lists)
    for layer_idx in range(1, 4):
        layer = rv[layer_idx]
        for i, val in enumerate(layer):
            col = f'router_vector_{layer_idx}_{i}'
            row[col] = val

    dataset.append(row)

# 3. Sort dataset by type using our explicit order map
dataset_sorted = sorted(dataset, key=lambda r: type_order.get(r['type'], float('inf')))

# 4. Write sorted data to CSV
header = [
    'image_id', 'label', 'type',
    'router_vector_0',
    'router_vector_1_0', 'router_vector_1_1',
    'router_vector_2_0', 'router_vector_2_1', 'router_vector_2_2',
    'router_vector_3_0', 'router_vector_3_1', 'router_vector_3_2', 'router_vector_3_3'
]
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for row in dataset_sorted:
        writer.writerow(row)

print(f"Done! Sorted CSV written to {OUTPUT_CSV}")
