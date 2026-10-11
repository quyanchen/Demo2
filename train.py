import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src import BertForNER, NERCollator, NERConfig, NERDataset, NERMetric, NERTrainer, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/weibo_bert_base.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_swanlab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = NERConfig.from_json(args.config)
    for k in ("epochs", "batch_size", "learning_rate", "max_length", "patience", "seed"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    if args.device != "auto":
        cfg.device = args.device

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Device: {device} | Loading tokenizer: {cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, cache_dir=cfg.cache_dir, use_fast=True)

    train_dataset = NERDataset(cfg.train_file, tokenizer, max_length=cfg.max_length)
    val_dataset = NERDataset(cfg.dev_file, tokenizer, label2id=train_dataset.label2id, max_length=cfg.max_length)
    test_dataset = NERDataset(cfg.test_file, tokenizer, label2id=train_dataset.label2id, max_length=cfg.max_length)
    data_collator = NERCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=data_collator)
    print(f"Labels: {len(train_dataset.label2id)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = BertForNER(
        model_name_or_path=cfg.model_name_or_path,
        num_labels=len(train_dataset.label2id),
        dropout=cfg.dropout,
        cache_dir=cfg.cache_dir,
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id,
    )

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    metric = NERMetric(id2label=train_dataset.id2label, ignore_index=-100)

    trainer = NERTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        metric=metric,
        device=device,
        config=cfg,
        use_swanlab=not args.no_swanlab,
        tokenizer=tokenizer,
        test_loader=test_loader,
    )
    summary = trainer.fit()
    print(f"Best Val Micro-F1: {summary['best_dev_micro_f1']:.4f} (Epoch {summary['best_epoch']}) | Test Micro-F1: {summary['test_micro_f1']:.4f}")
    print(f"Outputs: {trainer.output_dir}")


if __name__ == "__main__":
    main()
