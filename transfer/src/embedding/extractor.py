import os
import json
from typing import Dict, Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_ckpt_path: str, tokenizer_name: str, device: str = 'cuda') -> (Any, Any):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt_path, output_hidden_states=True).to(device)
    model.eval()
    return tokenizer, model


def load_and_prepare_dataset(data_path: str) -> Dataset:
    if data_path[-4:] == '.csv':
        # Complete path given
        df = pd.read_csv(data_path)
    else:
        # Need to combine training and validation set
        df_train = pd.read_csv(f'{data_path}/training.csv')
        df_val = pd.read_csv(f'{data_path}/validation.csv')
        df = pd.concat([df_train, df_val], ignore_index=True)
    if 'sentence' not in df.columns or 'correction' not in df.columns:
        raise ValueError("Input CSV must contain 'sentence' and 'correction' columns.")
    df = df.rename(columns={'sentence': 'text', 'correction': 'target'})
    return Dataset.from_pandas(df)


def tokenize_dataset(hf: Dataset, tokenizer, batch_size: int = None) -> Dataset:
    def _tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)
    tokenized = hf.map(_tokenize, batched=True, batch_size=batch_size)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized


def extract_hidden_states(tokenized_hf: Dataset, model, tokenizer, batch_size: int, device: str) \
        -> Dict[str, np.ndarray]:
    def _extract(batch):
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs).hidden_states[-1]  # shape: (batch_size, seq_len, hidden_size)
        return {'hidden_state': outputs[:, 0].cpu().numpy()}  # CLS token only

    hf_hidden = tokenized_hf.map(_extract, batched=True, batch_size=batch_size, load_from_cache_file=False)
    X = np.stack(hf_hidden['hidden_state'])
    y = np.array(hf_hidden['label'])
    texts = hf_hidden['text']
    return {'embeddings': X, 'labels': y, 'texts': texts}


def save_results(results: Dict[str, Any], output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'embeddings.npy'), results['embeddings'])
    np.save(os.path.join(output_path, 'labels.npy'), results['labels'])
    out_jsonl = os.path.join(output_path, 'texts.jsonl')
    with open(out_jsonl, 'w') as f:
        for item in results['texts']:
            f.write(json.dumps({'text': item}) + '\n')


def extract_cls_embeddings(model_ckpt_path: str, tokenizer_name: str, data_path: str, output_path: str,
                           batch_size: int = 256, device: str = 'cuda'):
    tokenizer, model = load_model_and_tokenizer(model_ckpt_path, tokenizer_name, device)
    hf = load_and_prepare_dataset(data_path)
    tokenized = tokenize_dataset(hf, tokenizer)
    results = extract_hidden_states(tokenized, model, tokenizer, batch_size, device)
    save_results(results, output_path)
    return results


def extract_cls_embeddings_from_samples(model_ckpt_path: str, tokenizer_name: str, samples: list, batch_size: int = 256,
                                        device: str = 'cuda'):
    tokenizer, model = load_model_and_tokenizer(model_ckpt_path, tokenizer_name, device)
    df = pd.DataFrame(samples)
    if 'sentence' not in df.columns or 'correction' not in df.columns:
        raise ValueError("Data must contain 'sentence' and 'correction' keys.")
    df = df.rename(columns={'sentence': 'text', 'correction': 'target'})
    hf = Dataset.from_pandas(df)
    tokenized = tokenize_dataset(hf, tokenizer)
    results = extract_hidden_states(tokenized, model, tokenizer, batch_size, device)
    return results
