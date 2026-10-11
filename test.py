import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src import BertForNER, NERCollator, NERConfig, NERDataset, NERMetric, get_device, print_metrics_table, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    device = get_device(args.device)

    print(f"Loading model from: {checkpoint_dir}")
    model = BertForNER.from_pretrained(checkpoint_dir, device=device)
    cfg = NERConfig.from_json(checkpoint_dir / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)

    if args.test_file is not None:
        cfg.test_file = args.test_file
    cfg.batch_size = args.batch_size

    test_dataset = NERDataset(cfg.test_file, tokenizer, label2id=model.label2id, max_length=cfg.max_length)
    data_collator = NERCollator(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=data_collator)
    print(f"Evaluating test set ({len(test_dataset)} samples)...")

    metric = NERMetric(id2label=model.id2label, ignore_index=-100)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            metric.update(outputs["logits"], batch["labels"])

    results = metric.compute()
    print_metrics_table(results)

    out_file = checkpoint_dir / "test_result.json"
    save_json(results, out_file)
    print(f"Saved test results to: {out_file}")


if __name__ == "__main__":
    main()
