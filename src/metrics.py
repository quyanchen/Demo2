from collections import defaultdict


class NERMetric:
    def __init__(self, id2label, ignore_index=-100):
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.ignore_index = ignore_index
        self.entity_types = sorted({lbl[2:] for lbl in self.id2label.values() if lbl.startswith(("B-", "I-"))})
        self.reset()

    @staticmethod
    def extract_entities_from_bio(tags):
        entities = set()
        current_entity = None
        for idx, tag in enumerate(tags):
            if tag.startswith("I-") and current_entity is not None and current_entity[0] == tag[2:]:
                continue
            if current_entity is not None:
                entities.add((current_entity[0], current_entity[1], idx))
                current_entity = None
            if tag.startswith("B-"):
                current_entity = (tag[2:], idx)
        if current_entity is not None:
            entities.add((current_entity[0], current_entity[1], len(tags)))
        return entities

    @staticmethod
    def print_metrics_table(results):
        print(f"\n{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        for ent_type, vals in results["classes"].items():
            print(f"{ent_type:<16} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1']:>10.4f} {vals['support']:>10}")
        print("-" * 60)
        print(f"{'Micro Avg':<16} {results['micro_precision']:>10.4f} {results['micro_recall']:>10.4f} {results['micro_f1']:>10.4f} {results['total_entities']:>10}")
        print(f"{'Macro Avg':<16} {'-':>10} {'-':>10} {results['macro_f1']:>10.4f} {results['total_entities']:>10}\n")

    def reset(self):
        self.correct = defaultdict(int)
        self.pred_count = defaultdict(int)
        self.gold_count = defaultdict(int)

    def update(self, predictions, labels):
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)
        preds = predictions.detach().cpu().tolist()
        targets = labels.detach().cpu().tolist()

        for pred_seq, target_seq in zip(preds, targets):
            valid_pred_tags, valid_gold_tags = [], []
            for p_id, t_id in zip(pred_seq, target_seq):
                if t_id == self.ignore_index:
                    continue
                valid_pred_tags.append(self.id2label.get(p_id, "O"))
                valid_gold_tags.append(self.id2label.get(t_id, "O"))

            pred_entities = self.extract_entities_from_bio(valid_pred_tags)
            gold_entities = self.extract_entities_from_bio(valid_gold_tags)

            for ent_type, _, _ in pred_entities & gold_entities:
                self.correct[ent_type] += 1
            for ent_type, _, _ in pred_entities:
                self.pred_count[ent_type] += 1
            for ent_type, _, _ in gold_entities:
                self.gold_count[ent_type] += 1

    def compute(self):
        per_class_results = {}
        f1_list = []
        for ent_type in self.entity_types:
            c = self.correct[ent_type]
            p = self.pred_count[ent_type]
            g = self.gold_count[ent_type]

            prec = c / p if p > 0 else 0.0
            rec = c / g if g > 0 else 0.0
            f1 = (2 * c) / (p + g) if (p + g) > 0 else 0.0

            per_class_results[ent_type] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": g,
            }
            if g > 0 or p > 0:
                f1_list.append(f1)

        total_c = sum(self.correct.values())
        total_p = sum(self.pred_count.values())
        total_g = sum(self.gold_count.values())

        micro_prec = total_c / total_p if total_p > 0 else 0.0
        micro_rec = total_c / total_g if total_g > 0 else 0.0
        micro_f1 = (2 * total_c) / (total_p + total_g) if (total_p + total_g) > 0 else 0.0
        macro_f1 = (sum(f1_list) / len(f1_list)) if f1_list else 0.0

        return {
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "classes": per_class_results,
            "total_entities": total_g,
        }
