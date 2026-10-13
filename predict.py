import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src import BertForNER, NERConfig, NERMetric, get_device


class NERPredictor:
    def __init__(self, checkpoint_dir, device="auto"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = get_device(device)
        self.model = BertForNER.from_pretrained(self.checkpoint_dir, device=self.device)
        self.cfg = NERConfig.from_json(self.checkpoint_dir / "config.json")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, use_fast=True)

    def predict(self, text):
        if not text.strip():
            return []

        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_length,
        )

        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            pred_ids = self.model(**inputs)["logits"][0].argmax(dim=-1).cpu().tolist()

        word_ids = encoding.word_ids(batch_index=0)
        char_tags = ["O"] * len(chars)
        current_word_idx = None
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != current_word_idx and word_id < len(chars):
                current_word_idx = word_id
                char_tags[word_id] = self.model.id2label[pred_ids[token_idx]]

        extracted = NERMetric.extract_entities_from_bio(char_tags)
        results = []
        for ent_type, start, end in sorted(extracted, key=lambda x: x[1]):
            results.append({"entity": text[start:end], "type": ent_type, "start": start, "end": end})
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = NERPredictor(checkpoint_dir=args.checkpoint_dir, device=args.device)

    if args.text:
        entities = predictor.predict(args.text)
        print(f"Input: {args.text}")
        print("Entities:")
        if not entities:
            print("  (None)")
        for item in entities:
            print(f"  - [{item['entity']}] -> {item['type']} ({item['start']}:{item['end']})")
    else:
        while True:
            try:
                user_input = input(">> ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                entities = predictor.predict(user_input)
                print(f"Entities: {entities}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
