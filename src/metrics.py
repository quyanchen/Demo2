from collections import defaultdict


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


def print_metrics_table(results):
    print(f"\n{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for ent_type, vals in results["classes"].items():
        print(f"{ent_type:<16} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1']:>10.4f} {vals['support']:>10}")
    print("-" * 60)
    print(f"{'Micro Avg':<16} {results['micro_precision']:>10.4f} {results['micro_recall']:>10.4f} {results['micro_f1']:>10.4f} {results['total_entities']:>10}")
    print(f"{'Macro Avg':<16} {'-':>10} {'-':>10} {results['macro_f1']:>10.4f} {results['total_entities']:>10}\n")


class NERMetric:
    def __init__(self, id2label, ignore_index=-100):
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.ignore_index = ignore_index
        self.entity_types = sorted({lbl[2:] for lbl in self.id2label.values() if lbl.startswith(("B-", "I-"))})
        self.reset()

    def reset(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

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

            pred_entities = extract_entities_from_bio(valid_pred_tags)
            gold_entities = extract_entities_from_bio(valid_gold_tags)
            for ent_type, _, _ in pred_entities & gold_entities:
                self.tp[ent_type] += 1
            for ent_type, _, _ in pred_entities - gold_entities:
                self.fp[ent_type] += 1
            for ent_type, _, _ in gold_entities - pred_entities:
                self.fn[ent_type] += 1

    def compute(self):
        per_class_results = {}
        f1_list = []
        for ent_type in self.entity_types:
            tp, fp, fn = self.tp[ent_type], self.fp[ent_type], self.fn[ent_type]
            support = tp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_results[ent_type] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
            if support > 0 or fp > 0:
                f1_list.append(f1)

        total_tp = sum(self.tp.values())
        total_fp = sum(self.fp.values())
        total_fn = sum(self.fn.values())
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        macro_f1 = (sum(f1_list) / len(f1_list)) if f1_list else 0.0

        return {
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "classes": per_class_results,
            "total_entities": total_tp + total_fn,
        }

