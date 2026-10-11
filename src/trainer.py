from datetime import datetime
from pathlib import Path
import swanlab
import torch
from tqdm import tqdm

from .metrics import print_metrics_table
from .utils import save_json


class NERTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        metric,
        device,
        config,
        use_swanlab=True,
        tokenizer=None,
        test_loader=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        self.cfg = config
        self.tokenizer = tokenizer
        self.use_swanlab = use_swanlab

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.cfg.output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.output_dir = self.output_dir.as_posix()
        self.cfg.to_json(self.output_dir / "config.json")

        self.best_val_f1 = float("-inf")
        self.best_epoch = -1
        self.patience_counter = 0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_micro_f1": [],
            "val_macro_f1": [],
        }

        if self.use_swanlab:
            swanlab.init(project=self.cfg.swanlab_project, experiment_name=self.cfg.experiment_name, config=self.cfg.to_dict())

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx + 1}/{self.cfg.epochs} [Train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs["loss"]

            loss.backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, data_loader, desc="Evaluating"):
        self.model.eval()
        self.metric.reset()
        total_loss = 0.0
        num_batches = len(data_loader)

        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            if outputs.get("loss") is not None:
                total_loss += outputs["loss"].item()

            self.metric.update(outputs["logits"], batch["labels"])

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, self.metric.compute()

    def save_checkpoint(self):
        self.model.save_pretrained(self.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.output_dir)

    def fit(self):
        print(f"Start training: epochs={self.cfg.epochs}, patience={self.cfg.patience}")

        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_results = self.evaluate(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")

            val_micro_f1 = val_results["micro_f1"]
            val_macro_f1 = val_results["macro_f1"]
            val_p = val_results["micro_precision"]
            val_r = val_results["micro_recall"]

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_precision"].append(val_p)
            self.history["val_recall"].append(val_r)
            self.history["val_micro_f1"].append(val_micro_f1)
            self.history["val_macro_f1"].append(val_macro_f1)

            if self.use_swanlab:
                swanlab.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_precision": val_p,
                    "val_recall": val_r,
                    "val_micro_f1": val_micro_f1,
                    "val_macro_f1": val_macro_f1,
                    "epoch": epoch + 1,
                })

            print(
                f"Epoch {epoch + 1:02d}/{self.cfg.epochs:02d} | "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"P: {val_p:.4f} | R: {val_r:.4f} | "
                f"micro_f1: {val_micro_f1:.4f} | macro_f1: {val_macro_f1:.4f}"
            )

            if val_micro_f1 > self.best_val_f1:
                self.best_val_f1 = val_micro_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint()
                print(f"  saved best checkpoint (micro_f1={val_micro_f1:.4f})")
            else:
                self.patience_counter += 1
                print(f"  no improvement ({self.patience_counter}/{self.cfg.patience})")
                if self.cfg.patience > 0 and self.patience_counter >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    break

        summary = {
            "best_epoch": self.best_epoch,
            "best_dev_micro_f1": self.best_val_f1,
            "history": self.history,
        }

        if self.test_loader is not None:
            best_weights = self.output_dir / "pytorch_model.bin"
            if best_weights.exists():
                state_dict = torch.load(best_weights, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            print(f"\nEvaluating best checkpoint (Epoch {self.best_epoch}) on test set...")
            test_loss, test_results = self.evaluate(self.test_loader, desc="Testing")
            print_metrics_table(test_results)
            save_json(test_results, self.output_dir / "test_result.json")
            summary["test_loss"] = round(test_loss, 4)
            summary["test_precision"] = test_results["micro_precision"]
            summary["test_recall"] = test_results["micro_recall"]
            summary["test_micro_f1"] = test_results["micro_f1"]
            summary["test_macro_f1"] = test_results["macro_f1"]
            if self.use_swanlab:
                swanlab.log({
                    "test_loss": test_loss,
                    "test_precision": test_results["micro_precision"],
                    "test_recall": test_results["micro_recall"],
                    "test_micro_f1": test_results["micro_f1"],
                    "test_macro_f1": test_results["macro_f1"],
                })

        save_json(summary, self.output_dir / "train_summary.json")

        if self.use_swanlab:
            swanlab.finish()

        return summary
