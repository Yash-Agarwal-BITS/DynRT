# DynRT/input/__init__.py

# Only import your unified JSON‐based loader and the tokenizer hooks
from .json_mmsd_loader import JSONMMSDLoader
from .requires         import get_tokenizer_roberta, get_tokenizer_bert

# Instantiate three roles: text, img, and label
_loaders = [
    JSONMMSDLoader("text"),
    JSONMMSDLoader("img"),
    JSONMMSDLoader("label"),
]

# These get passed into each loader during prepare()
_requires = {
    "tokenizer_roberta": get_tokenizer_roberta,
    "tokenizer_bert":    get_tokenizer_bert,
}

# Build the name→loader lookup map automatically
_loadermap = { loader.name: loader for loader in _loaders }
