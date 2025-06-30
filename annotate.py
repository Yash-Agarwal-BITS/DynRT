import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_JSON    = "input/text_json_final/train.json"
IMG_DIR      = "dataset_image"
OUTPUT_FILE  = "annotations.json"

N_NON        = 30     # non-sarcastic to auto-insert
N_SARC_TOTAL = 200    # sarcastic to review
# ────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def sample(items, n):
    return random.sample(items, min(n, len(items)))

def img_path(img_id):
    for ext in ("jpg", "png"):
        p = os.path.join(IMG_DIR, f"{img_id}.{ext}")
        if os.path.exists(p):
            return p
    return None

def show(img_file, caption=""):
    img = Image.open(img_file)
    plt.imshow(img); plt.axis("off"); plt.title(caption)
    plt.show(block=False)

def annotate():
    random.seed(42)

    data        = load_json(TEST_JSON)
    non_pool    = [e for e in data if e["label"] == 0]
    sarc_pool   = [e for e in data if e["label"] == 1]

    non_samples  = sample(non_pool,  N_NON)
    sarc_samples = sample(sarc_pool, N_SARC_TOTAL)

    annotations = []
    idx = 1
    counts = {"O":0, "S":0, "I":0, "rejected":0}
    mapping = {"O":"Object", "S":"Sentiment", "I":"Situational"}

    # Auto-add non-sarcastic
    for ex in non_samples:
        annotations.append({
            "id": idx,
            "image_id": ex["image_id"],
            "text": ex.get("text", ex.get("caption", "")),
            "label": 0,
            "type": ""
        })
        idx += 1

    print(f"\nInserted {len(non_samples)} non-sarcastic examples.")
    print(f"Now annotating {len(sarc_samples)} sarcastic examples…")
    print("Instructions: O=Object, S=Sentiment, I=Situational, R=Reject\n")

    # Sarcastic loop
    for ex in sarc_samples:
        path = img_path(ex["image_id"])
        if not path:
            print(f"⚠️  Missing image {ex['image_id']} — skipped.")
            counts["rejected"] += 1
            continue

        show(path, ex.get("text", ex.get("caption", "")))
        accepted = counts["O"] + counts["S"] + counts["I"]
        print(f"[Accepted so far] O:{counts['O']}  S:{counts['S']}  I:{counts['I']}  (total {accepted})")

        choice = ""
        while choice not in ["O", "S", "I", "R"]:
            choice = input("Choice (O/S/I/R) → ").strip().upper()

        plt.close()

        if choice == "R":
            counts["rejected"] += 1
            continue

        annotations.append({
            "id": idx,
            "image_id": ex["image_id"],
            "text": ex.get("text", ex.get("caption", "")),
            "label": 1,
            "type": mapping[choice]
        })
        idx += 1
        counts[choice] += 1

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(annotations, f, indent=2)

    # Summary
    print("\n=== Summary ===")
    print(f"Non-sarcastic saved  : {len(non_samples)}")
    print(f"Sarcastic accepted   : O:{counts['O']}  S:{counts['S']}  I:{counts['I']}")
    print(f"Sarcastic rejected   : {counts['rejected']}")
    print(f"Total written to {OUTPUT_FILE}: {len(annotations)}\n")

if __name__ == "__main__":
    annotate()