"""
mk/fine_tune/finetune.py
==============================================================================
FINE-TUNE a multilingual transformer to classify proverbs into the Kuusi
typology. This is the first method that actually CHANGES the model's weights
using your labels (everything before used frozen, off-the-shelf encoders).

Run it TWICE to compare the two models you asked for:
    python mk/fine_tune/finetune.py --model xlmr   --level theme
    python mk/fine_tune/finetune.py --model mbert  --level theme
    (xlmr  = xlm-roberta-base ;  mbert = bert-base-multilingual-cased)

WHAT IT DOES, STEP BY STEP
--------------------------
  1. Loads the example proverbs from the Kuusi CSV (text + gold codes + type id).
  2. Splits them with leak-free cross-validation: grouped by kuusi_id (so a type's
     translated variants never sit in both train and test) and stratified by theme.
  3. For each fold: loads the base transformer, adds a classification head sized to
     the number of classes, trains it on the proverb text for a few epochs, then
     predicts the held-out fold.
  4. Reports accuracy at the chosen level (theme / main / subgroup), mean +/- std
     across folds, and saves per-proverb predictions + a summary.

WHY THIS MIGHT BREAK THE 0.33 CEILING (and might not)
-----------------------------------------------------
Frozen encoders organize proverbs by topic; the typology organizes them by
abstract logic. Fine-tuning re-shapes the model so "same Kuusi category" becomes
the thing it encodes. That directly targets the mismatch. BUT if the label simply
isn't recoverable from the text, even a fine-tuned model will stall near 0.33.
Decision rule: if theme accuracy clears ~0.45, trained models help and we extend
to main/subgroup. If it stalls near 0.33, the ceiling is real and the path becomes
human-in-the-loop.

DEFAULTS: start with --level theme (13 classes, ~400 examples each = enough data).
Do NOT start with subgroup (316 classes, ~16 each = too thin to learn).

HARDWARE
--------
Needs an NVIDIA GPU to be practical. The script auto-detects CUDA and uses it; on
CPU it still runs but will be very slow. Lower --batch if you hit out-of-memory.

INPUT     mk/kuusi_proverb_types_clean.csv     (read from one level up)
OUTPUTS   mk/fine_tune/results/finetune_<model>_<level>_predictions.csv
          mk/fine_tune/results/finetune_<model>_<level>_summary.csv

INSTALL   pip install torch transformers scikit-learn pandas numpy
          (install the CUDA build of torch from https://pytorch.org for GPU)
==============================================================================
"""

import os                                              # paths
import argparse                                        # command-line flags
import numpy as np                                     # arrays / math
import pandas as pd                                    # tables
# NOTE: torch + transformers are imported INSIDE the training functions, so this
# file can be inspected / its data pipeline tested without those heavy libraries.

SEED = 42                                              # reproducibility

HERE = os.path.dirname(os.path.abspath(__file__))      # .../mk/fine_tune
MK   = os.path.dirname(HERE)                            # .../mk
KUUSI_CSV = os.path.join(MK, "kuusi_proverb_types_clean.csv")  # source data
RESULTS_DIR = os.path.join(HERE, "results")            # output folder

# Short names -> HuggingFace model ids. Both are multilingual (your data spans 13
# languages), so this is a fair BERT-vs-XLM-R comparison.
PRESETS = {
    "xlmr":  "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}

# The CSV columns that hold example proverbs (same exploding logic as 01/03).
EXAMPLE_COLS = [
    "proverb_variant_1", "proverb_variant_2", "proverb_variant_3",
    "proverb_variant_4", "proverb_variant_extra", "proverb_primary_en",
]

# Which gold column each --level uses as the label.
LEVEL_COL = {"theme": "theme", "main": "main_class", "subgroup": "subgroup_code"}


# =============================================================================
# DATA: explode the CSV into one row per example proverb (text + label + group)
# =============================================================================
def load_examples(level, kuusi_csv=KUUSI_CSV):
    """Return a DataFrame with columns: text, label, theme, kuusi_id.
    'label' is the gold code at the requested level; 'theme' is kept separately
    so we can stratify folds by theme; 'kuusi_id' is the grouping key."""
    df = pd.read_csv(kuusi_csv, encoding="utf-8-sig", dtype=str).fillna("")  # read source
    label_col = LEVEL_COL[level]                        # which column is the label
    rows = []                                           # collected example rows
    for _, r in df.iterrows():                          # walk every Kuusi type
        seen = set()                                    # de-dup identical strings within the type
        for col in EXAMPLE_COLS:                        # each example column
            t = r[col].strip()                          # the proverb text
            if t and t not in seen:                     # non-empty + not already added
                seen.add(t)                             # remember it
                rows.append(dict(text=t,                # the proverb
                                 label=r[label_col],    # its gold code at this level
                                 theme=r["theme"],      # its theme (for stratifying folds)
                                 kuusi_id=r["kuusi_id"]))  # its type id (for grouping)
    out = pd.DataFrame(rows)                            # assemble the table
    print(f"loaded {len(out)} example proverbs | level={level} | "
          f"{out['label'].nunique()} classes")
    return out


# =============================================================================
# FOLDS: leak-free cross-validation indices (group by kuusi_id, stratify by theme)
# =============================================================================
def make_folds(data, n_folds, seed=SEED):
    """Yield (train_idx, test_idx) so a type's variants never split across the two,
    and every fold sees every theme. Same idea as 02's evaluation."""
    try:
        from sklearn.model_selection import StratifiedGroupKFold     # preferred
        sp = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(sp.split(data["text"], data["theme"], groups=data["kuusi_id"])), \
               "StratifiedGroupKFold(group=kuusi_id, stratify=theme)"
    except Exception:
        from sklearn.model_selection import GroupKFold                # fallback
        sp = GroupKFold(n_splits=n_folds)
        return list(sp.split(data["text"], data["theme"], groups=data["kuusi_id"])), \
               "GroupKFold(group=kuusi_id)"


# =============================================================================
# TRAIN + PREDICT one fold  (this is where torch / transformers are used)
# =============================================================================
def train_fold(model_id, tr_text, tr_y, te_text, n_labels, args, device):
    """Fine-tune the base model on (tr_text -> tr_y) and return predicted label-ids
    for te_text. tr_y are integer class ids (0..n_labels-1)."""
    import torch                                        # deep-learning library
    from torch.utils.data import TensorDataset, DataLoader            # batching
    from transformers import (AutoTokenizer,            # text -> token ids
                              AutoModelForSequenceClassification,      # model + head
                              get_linear_schedule_with_warmup)        # LR schedule

    torch.manual_seed(SEED)                             # reproducible training

    tok = AutoTokenizer.from_pretrained(model_id)       # load the matching tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(       # load base model...
        model_id, num_labels=n_labels).to(device)       # ...with an n_labels-way head, on GPU/CPU

    # Turn text into padded token-id tensors (proverbs are short, so max_len is small).
    tr_enc = tok(list(tr_text), truncation=True, padding=True,
                 max_length=args.max_len, return_tensors="pt")
    te_enc = tok(list(te_text), truncation=True, padding=True,
                 max_length=args.max_len, return_tensors="pt")

    # A dataset of (input_ids, attention_mask, label) for the training rows.
    train_ds = TensorDataset(tr_enc["input_ids"], tr_enc["attention_mask"],
                             torch.tensor(tr_y, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)   # shuffle each epoch

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)            # optimizer
    total_steps = len(loader) * args.epochs                              # total gradient steps
    sched = get_linear_schedule_with_warmup(                            # warm up then decay LR
        optim, int(0.1 * total_steps), total_steps)
    use_amp = (device == "cuda")                        # mixed precision only helps on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # scales loss to avoid fp16 underflow

    model.train()                                       # put model in training mode
    for epoch in range(args.epochs):                    # several passes over the data
        for ids, mask, y in loader:                     # each mini-batch
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)  # move to device
            optim.zero_grad()                           # clear old gradients
            with torch.cuda.amp.autocast(enabled=use_amp):                # fp16 where safe
                out = model(input_ids=ids, attention_mask=mask, labels=y) # forward + loss
                loss = out.loss                         # cross-entropy loss (computed by the model)
            scaler.scale(loss).backward()               # backprop (scaled)
            scaler.step(optim)                          # optimizer step
            scaler.update()                             # update the loss scaler
            sched.step()                                # advance the LR schedule
        print(f"      epoch {epoch+1}/{args.epochs} done")

    # --- predict the held-out fold ---
    model.eval()                                        # evaluation mode (no dropout)
    preds = []                                          # collected predictions
    with torch.no_grad():                               # no gradients needed for inference
        for i in range(0, len(te_text), args.batch):    # iterate test rows in batches
            ids = te_enc["input_ids"][i:i+args.batch].to(device)         # batch of ids
            mask = te_enc["attention_mask"][i:i+args.batch].to(device)   # batch of masks
            with torch.cuda.amp.autocast(enabled=use_amp):               # fp16 inference
                logits = model(input_ids=ids, attention_mask=mask).logits  # class scores
            preds.append(logits.argmax(-1).cpu().numpy())                # pick top class
    del model                                           # free the model...
    if use_amp:                                         # ...and GPU memory before the next fold
        import torch as _t; _t.cuda.empty_cache()
    return np.concatenate(preds)                        # predicted label-ids for the test fold


# =============================================================================
# CROSS-VALIDATION DRIVER
# =============================================================================
def run(args):
    import torch                                        # only to detect the device here
    from sklearn.preprocessing import LabelEncoder      # map codes <-> integer ids

    device = "cuda" if torch.cuda.is_available() else "cpu"            # pick hardware
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be slow. Lower --epochs to test.")

    model_id = PRESETS.get(args.model, args.model)      # resolve preset or use raw HF id
    data = load_examples(args.level)                    # load + explode the proverbs

    le = LabelEncoder().fit(data["label"])              # fit label<->id mapping on ALL labels
    y_all = le.transform(data["label"])                 # integer id for every proverb
    n_labels = len(le.classes_)                         # number of classes at this level

    folds, split_name = make_folds(data, args.folds)    # leak-free CV splits
    print(f"\nmodel={model_id} | level={args.level} | classes={n_labels} | "
          f"device={device}\nCV: {split_name}, folds={args.folds}\n")

    texts = data["text"].values                         # all proverb strings
    rows, fold_acc = [], []                             # collected predictions + per-fold acc
    for f, (tr, te) in enumerate(folds):                # each fold
        # SAFETY: prove no type leaks across train/test.
        assert not (set(data["kuusi_id"].values[tr]) & set(data["kuusi_id"].values[te])), "LEAK"
        if args.max_train:                              # optional smoke-test cap (speed)
            tr = tr[:args.max_train]
        print(f"  fold {f}: train={len(tr)} test={len(te)}")
        pred_ids = train_fold(model_id, texts[tr], y_all[tr], texts[te],
                              n_labels, args, device)   # train + predict
        pred_codes = le.inverse_transform(pred_ids)     # ids back to Kuusi codes
        true_codes = data["label"].values[te]           # gold codes for the test fold
        acc = float(np.mean(pred_codes == true_codes))  # fold accuracy
        fold_acc.append(acc)                            # remember it
        print(f"  fold {f}: accuracy = {acc:.3f}\n")
        for j, i in enumerate(te):                       # store per-proverb predictions
            rows.append(dict(fold=f, text=texts[i],
                             true=true_codes[j], pred=pred_codes[j],
                             correct=int(pred_codes[j] == true_codes[j])))

    # --- report ---
    mean, std = float(np.mean(fold_acc)), float(np.std(fold_acc))       # aggregate
    print("================ FINE-TUNE RESULT ================")
    print(f"  {args.model}  level={args.level}  accuracy = {mean:.3f} +/- {std:.3f}")
    print(f"  (compare: pure-kNN theme=0.332 | best definition theme=0.301)")

    # --- save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{args.model}_{args.level}"                  # filename tag per model+level
    pred_path = os.path.join(RESULTS_DIR, f"finetune_{tag}_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding="utf-8-sig")
    summ_path = os.path.join(RESULTS_DIR, f"finetune_{tag}_summary.csv")
    pd.DataFrame([dict(model=args.model, level=args.level,
                       mean_acc=round(mean, 4), std_acc=round(std, 4))]
                 ).to_csv(summ_path, index=False)
    print(f"saved {pred_path}")
    print(f"saved {summ_path}")


def main():
    ap = argparse.ArgumentParser(description="Fine-tune XLM-R / mBERT for Kuusi classification.")
    ap.add_argument("--model", default="xlmr", help="xlmr | mbert | any HuggingFace id")
    ap.add_argument("--level", default="theme", choices=["theme", "main", "subgroup"],
                    help="which level to classify (start with theme)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--epochs", type=int, default=4, help="training passes per fold")
    ap.add_argument("--batch", type=int, default=16, help="batch size (lower if out-of-memory)")
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    ap.add_argument("--max_len", type=int, default=96, help="max tokens per proverb")
    ap.add_argument("--max_train", type=int, default=0,
                    help="cap training rows per fold for a fast smoke test (0 = use all)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()