from pathlib import Path
import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id=None, max_length=128):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        samples = self.load_data(self.file_path)
        self.label2id = self._build_label2id(samples) if label2id is None else dict(label2id)
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.features = [self._encode_sample(words, tags) for words, tags in samples]

    @staticmethod
    def load_data(file_path):
        samples = []
        words, tags = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        samples.append((words, tags))
                        words, tags = [], []
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    words.append(parts[0])
                    tags.append(parts[-1])
        if words:
            samples.append((words, tags))
        return samples

    @staticmethod
    def _build_label2id(samples):
        unique_labels = {tag for _, tags in samples for tag in tags}
        sorted_labels = ["O"] + sorted(unique_labels - {"O"})
        return {label: idx for idx, label in enumerate(sorted_labels)}

    def _encode_sample(self, words, tags):
        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=self.max_length)
        aligned_labels = []
        prev_word_idx = None
        for word_id in encoding.word_ids():
            if word_id is None or word_id == prev_word_idx:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label2id[tags[word_id]])
            prev_word_idx = word_id

        input_ids = encoding["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", [0] * len(input_ids)),
            "labels": aligned_labels,
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class NERCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, token_type_ids, labels = [], [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            token_type_ids.append(f["token_type_ids"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
