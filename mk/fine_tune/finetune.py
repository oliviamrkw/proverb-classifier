"""
mk/fine_tune/finetune.py
==============================================================================
FINE-TUNE a multilingual transformer (mBERT / XLM-R) to classify proverbs into
the Kuusi typology, with leak-free cross-validation.

WHAT'S NEW IN THIS VERSION
  1. CLASS-WEIGHTED LOSS  -> stops the collapse-to-one-class failure XLM-R hit.
     Rare themes get a heavier loss, so the model can't win by always guessing "H".
  2. SAVED PROBABILITIES  -> the predictions file now includes prob_<CODE> columns
     (the model's confidence for every theme), which the ensemble/fusion step needs.
  3. SPEED  -> shorter max_len (proverbs are ~6 words), per-batch dynamic padding,
     fp16, cudnn autotune, and an optional --freeze_layers lever. See "SPEED" below.

Run it for each model you want to compare:
    python mk/fine_tune/finetune.py --model mbert --level theme
    python mk/fine_tune/finetune.py --model xlmr  --level theme

SPEED knobs (biggest first):
  --max_len 48     proverbs are tiny (p99 = 21 tokens), so 48 is plenty. (was 96)
  --freeze_layers 6  freeze the bottom 6 transformer layers + embeddings; trains
                     ~1.5-2x faster, usually tiny accuracy change on small data.
  --batch 32       with the short max_len you can fit a bigger batch on 6 GB.
  --folds 3        fewer CV folds while experimenting (use 5 for the final number).
  Also: --model distilbert-base-multilingual-cased is a ~2x faster drop-in.
  NOTE: the #1 cause of slowness is running on CPU. This prints the device on
  startup -- if it says cpu, fix your CUDA torch install before anything else.

INPUT     mk/kuusi_proverb_types_clean.csv   (one level up)
OUTPUTS   mk/fine_tune/results/finetune_<model>_<level>_predictions.csv  (+ probs)
          mk/fine_tune/results/finetune_<model>_<level>_summary.csv
INSTALL   pip install torch transformers scikit-learn pandas numpy
==============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
# torch + transformers are imported inside train_fold(), so the data pipeline can
# be inspected without them.

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
MK   = os.path.dirname(HERE)
KUUSI_CSV = os.path.join(MK, "kuusi_proverb_types_clean.csv")
RESULTS_DIR = os.path.join(HERE, "results")

PRESETS = {"xlmr": "xlm-roberta-base", "mbert": "bert-base-multilingual-cased"}
EXAMPLE_COLS = [
    "proverb_variant_1", "proverb_variant_2", "proverb_variant_3",
    "proverb_variant_4", "proverb_variant_extra", "proverb_primary_en",
]
LEVEL_COL = {"theme": "theme", "main": "main_class", "subgroup": "subgroup_code"}


# =============================================================================
# DATA (unchanged): explode the CSV into one row per example proverb
# =============================================================================
def load_examples(level, kuusi_csv=KUUSI_CSV):
    df = pd.read_csv(kuusi_csv, encoding="utf-8-sig", dtype=str).fillna("")
    label_col = LEVEL_COL[level]
    rows = []
    for _, r in df.iterrows():
        seen = set()
        for col in EXAMPLE_COLS:
            t = r[col].strip()
            if t and t not in seen:
                seen.add(t)
                rows.append(dict(text=t, label=r[label_col],
                                 theme=r["theme"], kuusi_id=r["kuusi_id"]))
    out = pd.DataFrame(rows)
    print(f"loaded {len(out)} example proverbs | level={level} | {out['label'].nunique()} classes")
    return out


def make_folds(data, n_folds, seed=SEED):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sp = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(sp.split(data["text"], data["theme"], groups=data["kuusi_id"])), \
               "StratifiedGroupKFold(group=kuusi_id, stratify=theme)"
    except Exception:
        from sklearn.model_selection import GroupKFold
        sp = GroupKFold(n_splits=n_folds)
        return list(sp.split(data["text"], data["theme"], groups=data["kuusi_id"])), \
               "GroupKFold(group=kuusi_id)"


# These two helpers are defined at MODULE TOP LEVEL (not inside train_fold) so they
# can be pickled. On Windows, DataLoader workers are launched by pickling, and a class
# defined inside a function is not picklable -- that caused the "Can't get local object
# 'train_fold.<locals>.DS'" crash. Top-level = picklable = works on Windows too.
class _TextDS:                                          # holds raw (text, label) pairs
    def __init__(self, texts, labels): self.t, self.y = list(texts), list(labels)
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i], self.y[i]


class _Collate:                                         # tokenizes one batch (dynamic padding)
    def __init__(self, tok, max_len): self.tok, self.max_len = tok, max_len
    def __call__(self, batch):
        import torch
        txts = [b[0] for b in batch]; ys = [b[1] for b in batch]
        enc = self.tok(txts, truncation=True, padding=True, max_length=self.max_len,
                       return_tensors="pt")
        return enc, torch.tensor(ys, dtype=torch.long)


# =============================================================================
# TRAIN + PREDICT one fold  -- now returns PROBABILITIES, uses class weights
# =============================================================================
def train_fold(model_id, tr_text, tr_y, te_text, n_labels, args, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True               # SPEED: autotune kernels for fixed shapes

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_labels).to(device)

    # SPEED (optional): freeze embeddings + the bottom K transformer layers, so the
    # backward pass updates far fewer weights. Works for both BERT and XLM-R (their
    # base model exposes .embeddings and .encoder.layer).
    if args.freeze_layers > 0:
        base = model.base_model
        for p in base.embeddings.parameters():
            p.requires_grad = False
        for layer in base.encoder.layer[:args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        print(f"  froze embeddings + bottom {args.freeze_layers} layers")

    # CLASS WEIGHTS = inverse frequency -> prevents collapse to the majority class.
    counts = np.bincount(tr_y, minlength=n_labels).astype("float32")
    weights = counts.sum() / (n_labels * np.maximum(counts, 1))
    w = torch.tensor(weights, dtype=torch.float32, device=device)

    # tokenize PER BATCH (dynamic padding) so short batches aren't padded to a fixed
    # long length. The dataset/collate classes live at module top level (see above)
    # so DataLoader workers can be pickled on Windows.
    loader = DataLoader(_TextDS(tr_text, tr_y), batch_size=args.batch, shuffle=True,
                        collate_fn=_Collate(tok, args.max_len), num_workers=args.workers,
                        pin_memory=(device == "cuda"))

    # only optimize the parameters we didn't freeze
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total), total)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for ep in range(args.epochs):
        for enc, yb in loader:
            enc = {k: v.to(device) for k, v in enc.items()}; yb = yb.to(device)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**enc).logits
                loss = F.cross_entropy(logits, yb, weight=w)   # WEIGHTED loss
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update(); sched.step()
        print(f"      epoch {ep+1}/{args.epochs} done")

    # predict the held-out fold; keep the FULL softmax probabilities, not just argmax
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te_text), args.batch):
            chunk = list(te_text[i:i+args.batch])
            enc = tok(chunk, truncation=True, padding=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**enc).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())   # confidence per class
    del model
    if use_amp:
        torch.cuda.empty_cache()
    return np.concatenate(probs)                         # shape (n_test, n_labels)


# =============================================================================
# CROSS-VALIDATION DRIVER
# =============================================================================
def run(args):
    import torch
    from sklearn.preprocessing import LabelEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n*** DEVICE = {device} ***" + ("" if device == "cuda" else
          "  <-- CPU is the main reason this is slow; install CUDA torch."))

    model_id = PRESETS.get(args.model, args.model)
    data = load_examples(args.level)

    le = LabelEncoder().fit(data["label"])
    y_all = le.transform(data["label"])
    n_labels = len(le.classes_)
    classes = list(le.classes_)

    folds, split_name = make_folds(data, args.folds)
    print(f"model={model_id} | level={args.level} | classes={n_labels} | "
          f"max_len={args.max_len} | freeze={args.freeze_layers}\nCV: {split_name}, "
          f"folds={args.folds}\n")

    texts = data["text"].values
    rows, fold_acc = [], []
    for f, (tr, te) in enumerate(folds):
        assert not (set(data["kuusi_id"].values[tr]) & set(data["kuusi_id"].values[te])), "LEAK"
        if args.max_train:
            tr = tr[:args.max_train]
        print(f"  fold {f}: train={len(tr)} test={len(te)}")
        prob = train_fold(model_id, texts[tr], y_all[tr], texts[te], n_labels, args, device)
        pred_ids = prob.argmax(axis=1)                  # top class = highest probability
        pred_codes = le.inverse_transform(pred_ids)
        true_codes = data["label"].values[te]
        acc = float(np.mean(pred_codes == true_codes))
        fold_acc.append(acc)
        print(f"  fold {f}: accuracy = {acc:.3f}\n")
        for j, i in enumerate(te):
            rec = dict(fold=f, text=texts[i], true=true_codes[j], pred=pred_codes[j],
                       correct=int(pred_codes[j] == true_codes[j]))
            for c, cls in enumerate(classes):           # store the probability for every class
                rec[f"prob_{cls}"] = round(float(prob[j, c]), 4)
            rows.append(rec)

    mean, std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    pred_df = pd.DataFrame(rows)
    collapsed = pred_df["pred"].nunique() == 1          # sanity: did it degenerate?
    print("================ FINE-TUNE RESULT ================")
    print(f"  {args.model}  level={args.level}  accuracy = {mean:.3f} +/- {std:.3f}")
    if collapsed:
        print("  !! WARNING: model predicted ONE class only -- it collapsed. Try a lower "
              "--lr (e.g. 1e-5), more --epochs, or fewer --freeze_layers.")
    print(f"  (compare: pure-kNN theme=0.332 | best definition theme=0.301)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{args.model}_{args.level}"
    pred_df.to_csv(os.path.join(RESULTS_DIR, f"finetune_{tag}_predictions.csv"),
                   index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(model=args.model, level=args.level,
                       mean_acc=round(mean, 4), std_acc=round(std, 4))]
                 ).to_csv(os.path.join(RESULTS_DIR, f"finetune_{tag}_summary.csv"), index=False)
    print(f"saved predictions (with prob_<CODE> columns) + summary to {RESULTS_DIR}")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune mBERT / XLM-R for Kuusi classification.")
    ap.add_argument("--model", default="mbert", help="mbert | xlmr | any HuggingFace id")
    ap.add_argument("--level", default="theme", choices=["theme", "main", "subgroup"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32, help="raise if GPU memory allows")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=48, help="tokens per proverb (proverbs are tiny)")
    ap.add_argument("--freeze_layers", type=int, default=0,
                    help="freeze bottom N transformer layers for speed (try 6)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers (0 = safest, esp. on Windows)")
    ap.add_argument("--max_train", type=int, default=0, help="cap train rows per fold (smoke test)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()