from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from .utils import get_device, load_json, save_json


class BertForNER(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        num_labels,
        dropout=0.1,
        cache_dir=None,
        id2label=None,
        label2id=None,
        bert_config=None,
    ):
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.num_labels = num_labels
        self.id2label = id2label or {}
        self.label2id = label2id or {}

        if bert_config is not None:
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel.from_pretrained(self.model_name_or_path, cache_dir=cache_dir)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.bert.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(bert_outputs[0])
        logits = self.classifier(sequence_output)

        outputs = {"logits": logits}
        if labels is not None:
            outputs["loss"] = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), ignore_index=-100)
        return outputs

    def save_pretrained(self, save_directory):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")
        meta = {
            "model_name_or_path": self.model_name_or_path,
            "num_labels": self.num_labels,
            "dropout": self.dropout.p,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "label2id": self.label2id,
            "bert_config": self.bert.config.to_dict(),
        }
        save_json(meta, save_dir / "ner_model_config.json")

    @classmethod
    def from_pretrained(cls, save_directory, device="auto"):
        save_dir = Path(save_directory)
        meta = load_json(save_dir / "ner_model_config.json")
        id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}

        model = cls(
            model_name_or_path=meta["model_name_or_path"],
            num_labels=meta["num_labels"],
            dropout=meta.get("dropout", 0.1),
            id2label=id2label,
            label2id=meta.get("label2id", {}),
            bert_config=BertConfig.from_dict(meta["bert_config"]),
        )

        state_dict = torch.load(save_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(get_device(device))
        model.eval()
        return model
