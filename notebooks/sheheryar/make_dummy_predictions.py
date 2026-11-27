"""
Generate DUMMY predictions CSVs so the evaluation pipeline can be built before the real ones arrive.

    python notebooks/sheheryar/make_dummy_predictions.py

Writes one CSV per model to outputs/predictions/ (fixed schema, identical utterances and
true labels in identical order), a manifest that lets eval.py label these files as DUMMY,
and one deliberately misaligned CSV to tests/fixtures/ for the alignment-check test.

Real files from Ali (bert_finetuned, bilstm) and Davit (zero_shot, few_shot, balanced_few_shot)
replace these. This script never overwrites a CSV that is not one of its own dummies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as ev  # noqa: E402

SEED = 27  # workshop day

# Approximately MELD's official test distribution (2,610 rows).
CLASS_COUNTS = {"neutral": 1256, "joy": 402, "anger": 345, "surprise": 281,
                "sadness": 208, "disgust": 68, "fear": 50}

# Emotions that get mistaken for each other (symmetric weights).
SIMILAR = {("anger", "disgust"): 1.0, ("fear", "surprise"): 1.0, ("joy", "surprise"): 0.5,
           ("sadness", "fear"): 0.3, ("anger", "surprise"): 0.3, ("sadness", "anger"): 0.2}

# Per model: recall for each true emotion, how hard wrong answers fall into 'neutral'
# (models trained on imbalanced data dump errors there), and how hard they fall into similar emotions.
PROFILES = {
    "bert_finetuned": dict(
        recall=dict(neutral=0.80, joy=0.52, anger=0.44, surprise=0.52, sadness=0.30, disgust=0.12, fear=0.10),
        neutral_pull=6.0, similar_pull=3.0),
    "bilstm": dict(
        recall=dict(neutral=0.74, joy=0.38, anger=0.28, surprise=0.36, sadness=0.15, disgust=0.03, fear=0.02),
        neutral_pull=7.0, similar_pull=2.0),
    "zero_shot": dict(
        recall=dict(neutral=0.40, joy=0.55, anger=0.50, surprise=0.55, sadness=0.45, disgust=0.25, fear=0.30),
        neutral_pull=1.0, similar_pull=4.0),
    "few_shot": dict(
        recall=dict(neutral=0.50, joy=0.58, anger=0.50, surprise=0.56, sadness=0.45, disgust=0.18, fear=0.20),
        neutral_pull=1.6, similar_pull=4.0),
    "balanced_few_shot": dict(
        recall=dict(neutral=0.42, joy=0.56, anger=0.50, surprise=0.55, sadness=0.50, disgust=0.45, fear=0.48),
        neutral_pull=1.6, similar_pull=1.2),
}

# Synthetic lines (not from MELD), one pool per emotion. A numbered prefix makes every row unique.
TEMPLATES = {
    "anger": ["Are you kidding me right now?", "Get out of my apartment!", "That is so not fair!",
              "You read my diary?!", "I told you three times already!"],
    "disgust": ["Ew, that is disgusting.", "Oh gross, is that from last week?", "I can't even look at that.",
                "You put ketchup on that?"],
    "fear": ["What was that noise?", "I'm really scared she'll find out.", "Please don't make me go in there.",
             "What if the plane goes down?"],
    "joy": ["This is the best day ever!", "I got the job!", "Oh, I love you guys!", "We're going to Vegas!"],
    "neutral": ["So what time is the meeting?", "I'll grab the coffee.", "Okay.", "He said he'd call tomorrow.",
                "Is this seat taken?", "The keys are on the counter."],
    "sadness": ["I just really miss her.", "Nobody even remembered.", "I guess it's over then.",
                "He didn't even say goodbye."],
    "surprise": ["Wait, what?!", "You're getting married?!", "No way!", "Since when do you knit?"],
}


def make_truth(rng: np.random.Generator, repo_root: Path) -> pd.DataFrame:
    """Utterances + true labels. Uses the real official test rows once Mehwish's file exists."""
    official = ev.load_official_test(repo_root)
    if official is not None:
        print(f"Using the real official MELD test rows ({len(official)}) from data/processed/meld_clean.csv")
        return official
    labels = np.array([lab for lab, n in CLASS_COUNTS.items() for _ in range(n)])
    rng.shuffle(labels)
    utterances = [f"[dummy {i:04d}] {rng.choice(TEMPLATES[lab])}" for i, lab in enumerate(labels, 1)]
    print(f"Using {len(labels)} synthetic rows (data/processed/meld_clean.csv not in the repo yet)")
    return pd.DataFrame({"utterance": utterances, "true_label": labels})


def error_weights(profile: dict) -> np.ndarray:
    """W[t, p] = how likely a wrong answer for true emotion t lands on p."""
    n = len(ev.LABELS)
    w = np.ones((n, n))
    for (a, b), s in SIMILAR.items():
        i, j = ev.LABELS.index(a), ev.LABELS.index(b)
        w[i, j] += profile["similar_pull"] * s
        w[j, i] += profile["similar_pull"] * s
    w[:, ev.LABELS.index("neutral")] *= profile["neutral_pull"]
    np.fill_diagonal(w, 0)
    return w / w.sum(axis=1, keepdims=True)


def simulate(y_true: np.ndarray, profile: dict, rng: np.random.Generator) -> np.ndarray:
    weights = error_weights(profile)
    y_pred = y_true.copy()
    for i, label in enumerate(ev.LABELS):
        idx = np.flatnonzero(y_true == label)
        wrong = idx[rng.random(idx.size) >= profile["recall"][label]]
        y_pred[wrong] = rng.choice(ev.LABELS, size=wrong.size, p=weights[i])
    return y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--pred-dir", default=None, help="where to write the CSVs (default: outputs/predictions)")
    parser.add_argument("--fixtures-dir", default=None, help="where to write the misaligned file (default: tests/fixtures)")
    args = parser.parse_args()

    root = ev.find_repo_root(args.repo_root or Path(__file__).parent)
    pred_dir = Path(args.pred_dir) if args.pred_dir else root / "outputs" / "predictions"
    fixtures = Path(args.fixtures_dir) if args.fixtures_dir else root / "tests" / "fixtures"

    def shown(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    pred_dir.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)

    manifest_path = pred_dir / ev.DUMMY_MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    known_dummies = set(manifest.values())

    rng = np.random.default_rng(SEED)
    truth = make_truth(rng, root)
    y_true = truth["true_label"].to_numpy()

    written = {}
    for k, model in enumerate(ev.MODEL_NAMES):
        path = pred_dir / f"{model}_predictions.csv"
        if path.exists() and ev.file_sha256(path) not in known_dummies:
            print(f"SKIP  {path.name}: a real (non-dummy) file is already there, leaving it alone")
            continue
        y_pred = simulate(y_true, PROFILES[model], np.random.default_rng(SEED + k + 1))
        df = pd.DataFrame({"utterance": truth["utterance"], "true_label": y_true,
                           "predicted_label": y_pred, "model_name": model})[ev.COLUMNS]
        df.to_csv(path, index=False)
        manifest[path.name] = ev.file_sha256(path)
        written[model] = df
        r = ev.score(model, y_true, y_pred)
        print(f"wrote {shown(path)}  accuracy {ev.pct(r.accuracy):>6}  macro F1 {ev.pct(r.macro_f1):>6}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Deliberately misaligned copy: two neighbouring rows swapped, which is the kind of quiet
    # shuffle that still produces a clean-looking chart. The alignment check must reject it.
    source = written.get("bert_finetuned")
    if source is None:
        source = ev.read_predictions_csv(pred_dir / "bert_finetuned_predictions.csv")
    bad = source.copy()
    bad.iloc[[1233, 1234]] = bad.iloc[[1234, 1233]].to_numpy()
    bad_path = fixtures / "misaligned_bert_finetuned_predictions.csv"
    bad.to_csv(bad_path, index=False)
    print(f"wrote {shown(bad_path)}  (rows 1233 and 1234 swapped: must FAIL alignment)")


if __name__ == "__main__":
    main()
