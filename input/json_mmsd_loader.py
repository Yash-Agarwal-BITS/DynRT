# input/json_mmsd_loader.py

import json, os
import torch, numpy as np

class JSONMMSDLoader:
    def __init__(self, name):
        assert name in ("text", "img", "label")
        self.name = name
        self.splits = {}

    def prepare(self, inputs, opt):
        self.tokenizer = inputs.get("tokenizer_roberta", None)
        self.data_path = opt["data_path"]
        if self.name == "img":
            self.transform_image = opt["transform_image"]

        for split in ("train", "valid", "test"):
            path = os.path.join(self.data_path, f"{split}.json")
            with open(path, "r", encoding="utf-8") as f:
                self.splits[split] = json.load(f)

    def getlength(self, mode):
        return len(self.splits[mode])

    def get(self, output, mode, idx):
        sample = self.splits[mode][idx]

        if self.name == "text":
            toks = self.tokenizer(
                sample["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=100  # optional: match config len
            )
            output["text"] = toks["input_ids"].squeeze(0)
            output["text_mask"] = toks["attention_mask"].squeeze(0).bool()


        elif self.name == "img":
            stem = str(sample["image_id"])
            path = os.path.join(self.transform_image, stem + ".npy")
            arr = np.load(path)
            output["img"] = torch.from_numpy(arr)

        elif self.name == "label":
            output["label"] = torch.tensor(sample["label"], dtype=torch.long)
