# extract_router.py

import os
import json
import torch
import numpy as np
import argparse
from model.DynRT import build_DynRT
from transformers import RobertaTokenizer

def extract_alphas(opt):
    with open(opt.config, 'r') as f:
        config = json.load(f)
    print(f"Config loaded from: {opt.config}")
    
    model_opt = config['opt']['modelopt']
    device = torch.device('cuda:' + str(opt.gpu) if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_DynRT(model_opt, None).to(device)
    
    model_path = config['info']['test_on_checkpoint']
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Model weights loaded successfully from {model_path}")
    else:
        print(f"FATAL: Model weights not found at {model_path}.")
        return

    model.eval()

    try:
        with open('annotations.json', 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} samples from annotations.json")
    except FileNotFoundError:
        print("FATAL: annotations.json not found.")
        return

    tokenizer_path = config['opt']['dataloader']['requires']['tokenizer_roberta']['path']
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    
    results = []
    processed = 0
    skipped = 0
    
    for sample in annotations:
        image_id = str(sample.get('image_id'))  # Always use image_id for filename lookup
        text = sample['text']
        sarcasm_type = sample.get('sarcasm_type') if 'sarcasm_type' in sample else sample.get('type', 'N/A')

        # No filtering, process every entry
        processed += 1

        image_tensor_dir = config['opt']['dataloader']['loaders']['img']['transform_image']
        image_tensor_path = os.path.join(image_tensor_dir, image_id + '.npy')
        
        if not os.path.exists(image_tensor_path):
            print(f"Warning: Image tensor not found at {image_tensor_path}. Skipping sample.")
            continue
            
        img_tensor = torch.from_numpy(np.load(image_tensor_path)).unsqueeze(0).to(device)
        
        toks = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=model_opt['len']
        )
        txt_tensor = toks['input_ids'].to(device)
        mask_tensor = toks['attention_mask'].to(device).bool()  # [1, seq_len]

        model_input = {
            model_opt['input1']: txt_tensor,
            model_opt['input2']: img_tensor,
            model_opt['input3']: mask_tensor
        }

        with torch.no_grad():
            _result, _lang_emb, _img_emb, all_alphas = model(model_input)

        # Convert all router vectors to lists for JSON serialization
        router_vectors = [alpha.squeeze().cpu().numpy().tolist() for alpha in all_alphas]

        results.append({
            'image_id': image_id,
            'text': sample['text'],
            'label': sample['label'],
            'type': sample.get('type', ''),
            'router_vectors': router_vectors
        })
        
    print(f"Processed {processed} samples.")
    output_filename = 'router_output_vectors_tau(0.1).json'
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Extraction complete! Router output vectors saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/DynRT-test.json", help="Path to the model config file.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID to use.")
    opt = parser.parse_args()
    
    extract_alphas(opt)