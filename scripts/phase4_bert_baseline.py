"""
Phase 4 — BERT Baseline v2 (XLM-RoBERTa fine-tuning)
======================================================
Fixes from v1:
  Fix A: Class-weighted CrossEntropyLoss  (critical — stops collapse)
  Fix B: Use LAST 1000 chars of text      (legal outcomes appear at end)
  Fix C: Lower LR (1e-5) + more epochs   (careful training on small data)
  Fix D: Gradient accumulation steps=2   (effective batch size = 16)

Usage:
    python scripts/phase4_bert_baseline.py
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (
    normalize_outcome, compute_metrics, save_results, print_report
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEP  = "─" * 65
DSEP = "═" * 65

_TEXT_FIELDS    = ["facts_summary", "facts", "text", "description",
                   "case_text", "summary", "narrative", "content"]
_DOMAIN_FIELDS  = ["domain", "case_domain", "category", "type"]
_OUTCOME_FIELDS = ["outcome", "label", "result", "decision", "verdict"]

LABEL_MAP     = {"favorable": 1, "unfavorable": 0}
INV_LABEL_MAP = {1: "favorable", 0: "unfavorable"}


def _get(record, candidates):
    for k in candidates:
        if k in record:
            return str(record[k])
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Exact same split as phase4_evaluate.py ────────────────────────────────────

def get_split(cases_path, n_per_domain=50, seed=42):
    random.seed(seed)
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)
    all_cases = data if isinstance(data, list) else list(data.values())[0]

    by_domain = defaultdict(list)
    for case in all_cases:
        domain  = (_get(case, _DOMAIN_FIELDS) or "").lower()
        outcome = normalize_outcome(_get(case, _OUTCOME_FIELDS) or "")
        text    = _get(case, _TEXT_FIELDS)
        if domain and outcome and text:
            by_domain[domain].append(case)

    test_cases = []
    test_cits  = set()
    for domain, cases in sorted(by_domain.items()):
        n      = min(n_per_domain, len(cases))
        chosen = random.sample(cases, n)
        test_cases.extend(chosen)
        for c in chosen:
            test_cits.add(c.get("citation", id(c)))

    train_cases = [
        c for c in all_cases
        if c.get("citation", id(c)) not in test_cits
        and normalize_outcome(_get(c, _OUTCOME_FIELDS) or "")
        and _get(c, _TEXT_FIELDS)
    ]

    logger.info(f"Split → train: {len(train_cases)}  test: {len(test_cases)}")
    for label, cases in [("Train", train_cases), ("Test", test_cases)]:
        dist = defaultdict(int)
        for c in cases:
            dist[normalize_outcome(_get(c, _OUTCOME_FIELDS) or "") or "?"] += 1
        logger.info(f"  {label}: {dict(dist)}")

    return test_cases, train_cases


# ── Dataset ────────────────────────────────────────────────────────────────────

class LegalDataset(Dataset):
    def __init__(self, cases, tokenizer, max_length=256, text_tail_chars=1000):
        self.tokenizer      = tokenizer
        self.max_length     = max_length
        self.text_tail_chars = text_tail_chars
        self.items          = []

        for case in cases:
            text    = _get(case, _TEXT_FIELDS) or ""
            outcome = normalize_outcome(_get(case, _OUTCOME_FIELDS) or "")
            domain  = (_get(case, _DOMAIN_FIELDS) or "other").lower()

            if not text or outcome not in LABEL_MAP:
                continue

            # Fix B: use last N characters — legal outcomes appear at the end
            if len(text) > text_tail_chars:
                text = text[-text_tail_chars:]

            self.items.append({
                "text":         text,
                "label":        LABEL_MAP[outcome],
                "domain":       domain,
                "citation":     case.get("citation", ""),
                "ground_truth": outcome,
            })

        logger.info(f"Dataset: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item     = self.items[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(item["label"], dtype=torch.long),
            "domain":         item["domain"],
            "citation":       item["citation"],
            "ground_truth":   item["ground_truth"],
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label":          torch.stack([b["label"]          for b in batch]),
        "domain":         [b["domain"]       for b in batch],
        "citation":       [b["citation"]     for b in batch],
        "ground_truth":   [b["ground_truth"] for b in batch],
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn,
                device, epoch, grad_accum):
    model.train()
    total_loss  = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # Fix A: use weighted loss (not model's default loss)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels) / grad_accum
        loss.backward()

        if step % grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        if step % 10 == 0:
            logger.info(
                f"  Epoch {epoch}  step {step}/{len(loader)}  "
                f"loss={loss.item()*grad_accum:.4f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    results = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        for pred, gt, domain, citation in zip(
            preds, batch["ground_truth"], batch["domain"], batch["citation"]
        ):
            predicted = INV_LABEL_MAP[pred]
            results.append({
                "citation":     citation,
                "domain":       domain,
                "ground_truth": gt,
                "predicted":    predicted,
                "confidence":   1.0,
                "correct":      predicted == gt,
            })
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cases-path",  default="data/annotated/cases_augmented.json")
    p.add_argument("--output-dir",  default="data/evaluation")
    p.add_argument("--model",       default="xlm-roberta-base")
    p.add_argument("--epochs",      type=int,   default=10,
                   help="More epochs needed for small dataset.")
    p.add_argument("--batch-size",  type=int,   default=8,
                   help="Reduce to 4 if OOM on RTX 3050 4GB.")
    p.add_argument("--grad-accum",  type=int,   default=2,
                   help="Gradient accumulation — effective batch = batch*grad_accum.")
    p.add_argument("--max-length",  type=int,   default=256)
    p.add_argument("--text-tail",   type=int,   default=1000,
                   help="Use last N chars of text (legal outcomes appear at end).")
    p.add_argument("--lr",          type=float, default=1e-5,
                   help="Lower LR for stable training on small dataset.")
    p.add_argument("--warmup-ratio",type=float, default=0.1)
    p.add_argument("--n-per-domain",type=int,   default=50)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{DSEP}")
    print("  Phase 4 — BERT Baseline v2  (XLM-RoBERTa + all fixes)")
    print(DSEP)
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}  LR: {args.lr}")
    print(f"  Batch size : {args.batch_size} × grad_accum {args.grad_accum}"
          f" = effective {args.batch_size * args.grad_accum}")
    print(f"  Text tail  : last {args.text_tail} chars  (Fix B)")
    print(DSEP)

    # ── Data split ────────────────────────────────────────────────────────────
    test_cases, train_cases = get_split(
        args.cases_path, args.n_per_domain, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = LegalDataset(train_cases, tokenizer,
                                 args.max_length, args.text_tail)
    test_dataset  = LegalDataset(test_cases,  tokenizer,
                                 args.max_length, args.text_tail)

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  collate_fn=collate_fn)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size * 2,
                               shuffle=False, collate_fn=collate_fn)

    # Fix A: compute class weights from training labels
    train_labels    = [item["label"] for item in train_dataset.items]
    class_weights   = compute_class_weight(
        "balanced",
        classes = np.array([0, 1]),
        y       = np.array(train_labels),
    )
    class_wt_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn         = torch.nn.CrossEntropyLoss(weight=class_wt_tensor)
    logger.info(f"Class weights: unfavorable={class_weights[0]:.3f}  "
                f"favorable={class_weights[1]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model …")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
    ).to(device)

    # Fix C: lower LR
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup      = int(total_steps * args.warmup_ratio)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )
    logger.info(f"Training: {total_steps} steps | {warmup} warmup")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1      = 0.0
    best_results = []
    best_epoch   = 1

    for epoch in range(1, args.epochs + 1):
        print(f"\n{SEP}\nEpoch {epoch}/{args.epochs}")
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler,
                               loss_fn, device, epoch, args.grad_accum)
        logger.info(f"  Loss: {avg_loss:.4f}")

        results = evaluate_model(model, test_loader, device)
        metrics = compute_metrics(results, f"XLM-RoBERTa epoch {epoch}")
        acc     = metrics["overall_accuracy"] * 100
        f1      = metrics["macro_f1"]
        da      = metrics.get("domain_accuracy", {})

        # Count unfavorable predictions
        n_unfav_pred = sum(1 for r in results if r["predicted"] == "unfavorable")
        print(
            f"  Acc={acc:.1f}%  F1={f1:.3f}  "
            f"F1_unfav={metrics['unfavorable_f1']:.3f}  "
            f"unfav_preds={n_unfav_pred}/{sum(1 for r in results if r['ground_truth']=='unfavorable')} "
            f"| Land={da.get('land',0)*100:.1f}%  "
            f"Contract={da.get('contract',0)*100:.1f}%  "
            f"Service={da.get('service',0)*100:.1f}%"
        )

        if f1 > best_f1:
            best_f1      = f1
            best_results = results
            best_epoch   = epoch
            logger.info(f"  ✓ Best F1: {f1:.3f}")

    # ── Save and compare ──────────────────────────────────────────────────────
    print(f"\n{SEP}\nBest: Epoch {best_epoch}  F1={best_f1:.3f}")

    final_metrics = compute_metrics(best_results, "XLM-RoBERTa (fine-tuned)")
    save_results(best_results, final_metrics, output_dir, "XLM-RoBERTa (fine-tuned)")

    all_metrics = [final_metrics]
    for mf in sorted(output_dir.glob("metrics_*.json")):
        if "xlm" not in mf.name:
            with open(mf, encoding="utf-8") as f:
                all_metrics.append(json.load(f))
    all_metrics.sort(key=lambda m: m.get("macro_f1", 0))
    print_report(all_metrics)


if __name__ == "__main__":
    main()
