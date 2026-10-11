from dataclasses import asdict, dataclass
from .utils import load_json, save_json


@dataclass
class NERConfig:
    train_file: str = "data/weibo/train.txt"
    dev_file: str = "data/weibo/dev.txt"
    test_file: str = "data/weibo/test.txt"

    model_name_or_path: str = "google-bert/bert-base-chinese"
    cache_dir: str = r"D:\1Code\1Model\huggingface"
    dropout: float = 0.1

    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    grad_clip: float = 1.0
    seed: int = 42
    patience: int = 3
    device: str = "auto"

    output_dir: str = "outputs/weibo_bert_base"
    swanlab_project: str = "demo2-ner"
    experiment_name: str = "weibo_bert_base"

    @classmethod
    def from_json(cls, json_path):
        return cls(**load_json(json_path))

    def to_json(self, json_path):
        save_json(self.to_dict(), json_path)

    def to_dict(self):
        return asdict(self)
