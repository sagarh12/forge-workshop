"""
Forge Workshop 01 - evaluation pipeline (Sheheryar's block: "The Scoreboard").

notebooks/sheheryar/evaluation.ipynb imports this module. The pipeline:
  1. finds every outputs/predictions/*.csv (no hardcoded filenames, any subset of models works),
  2. runs the ALIGNMENT CHECK FIRST and refuses to score anything if a real file is misaligned,
  3. scores each model on the SHARED test rows: accuracy, macro F1, classification report,
     confusion matrix (LLM runs cover the first 100 test lines, trained models all 2,610, so
     every model is compared on the first 100; full-test-set scores are kept alongside),
  4. builds the scoreboard, the "accuracy lies" majority baseline and the master comparison chart.

Run it without the notebook:  python notebooks/sheheryar/eval.py
Works with pandas 2.x (Colab) and 3.x.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

# --------------------------------------------------------------------------- shared contract
# These come straight from the runbook's non-negotiable rules. Charts key off these strings.
COLUMNS = ["utterance", "true_label", "predicted_label", "model_name"]
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODEL_NAMES = ["bilstm", "bert_finetuned", "zero_shot", "few_shot", "balanced_few_shot"]

# The baseline always answers the most common emotion in MELD's TRAINING split. It is fixed here
# rather than read off the test labels, so the baseline never peeks at the test set.
MAJORITY_LABEL = "neutral"

# Every predictions file covers the whole official test split, except Davit's three LLM runs, which
# may instead be EXACTLY the first 100 official test lines, in order (saved with
# save_predictions(..., full_test_set=False)). Nothing else is accepted: a 40-line live demo, a
# 150-line run, or a 100-line BERT/BiLSTM file would quietly change what the scoreboard compares.
LLM_MODELS = ("zero_shot", "few_shot", "balanced_few_shot")
LLM_RUN_ROWS = 100

DISPLAY_NAMES = {
    "bilstm": "BiLSTM",
    "bert_finetuned": "BERT\n(fine-tuned)",
    "zero_shot": "LLM\nzero-shot",
    "few_shot": "LLM\nfew-shot",
    "balanced_few_shot": "LLM balanced\nfew-shot",
}
APPROACH = {  # Beat 2 of the story: train a model vs just ask an LLM
    "bilstm": "trained model",
    "bert_finetuned": "trained model",
    "zero_shot": "LLM prompt",
    "few_shot": "LLM prompt",
    "balanced_few_shot": "LLM prompt",
}

# Written by make_dummy_predictions.py. Any CSV whose hash is listed here is fake data.
DUMMY_MANIFEST = "_dummy_manifest.json"

# Forge Workshop 01 deck palette
NAVY, BLUE, LIGHT_BLUE, GREY, INK = "#0F1F4D", "#3D5BFF", "#9DB0FF", "#62676D", "#15171A"
WARN_RED = "#C62828"


class AlignmentError(Exception):
    """A predictions CSV does not line up with the others. Never catch and ignore this."""


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def display_name(model_name: str) -> str:
    return DISPLAY_NAMES.get(model_name, model_name)


def _banner(title: str, body: list[str]) -> str:
    return "\n".join(["", "=" * 78, title, "=" * 78, *body, "=" * 78])


# --------------------------------------------------------------------------- locating files
def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) to the folder holding forge_workshop_base.ipynb."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "forge_workshop_base.ipynb").exists():
            return folder
    for folder in [here, *here.parents]:
        if (folder / "outputs" / "predictions").is_dir():
            return folder
    raise FileNotFoundError(f"Could not find the forge-workshop repo root above {here}")


def find_prediction_files(pred_dir: str | Path) -> list[Path]:
    """Every *.csv in the predictions folder, sorted. No filenames are hardcoded."""
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(_banner(
            f"NO PREDICTIONS FOUND in {pred_dir}",
            ["Expected one CSV per model, saved with save_predictions() from the base notebook."]))
    return files


def read_predictions_csv(path: str | Path) -> pd.DataFrame:
    """Read every column as text, exactly as written: no "NA" -> NaN guessing, BOM tolerated."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        raise AlignmentError(_banner(f"CANNOT READ {Path(path).name}", [
            f"{type(err).__name__}: {err}",
            "Re-save it with save_predictions() from the base notebook.",
            "Don't open and re-save prediction files in Excel: it changes the encoding."])) from err


def load_predictions(pred_dir: str | Path) -> dict[str, pd.DataFrame]:
    """{filename: DataFrame} for every CSV present."""
    return {p.name: read_predictions_csv(p) for p in find_prediction_files(pred_dir)}


def load_official_test(repo_root: str | Path) -> pd.DataFrame | None:
    """The official MELD test rows (utterance, true_label) in official order, or None if
    Mehwish's data/processed/meld_clean.csv is not in the repo yet."""
    path = Path(repo_root) / "data" / "processed" / "meld_clean.csv"
    if not path.exists():
        return None
    meld = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in ("utterance", "label", "split") if c not in meld.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(meld.columns)}")
    # Same split logic as Step 6 of the base notebook, so the row order matches save_predictions().
    split = meld["split"].str.lower().replace({"val": "dev", "validation": "dev"})
    test = (
        meld.loc[split == "test", ["utterance", "label"]]
        .rename(columns={"label": "true_label"})
        .reset_index(drop=True)
    )
    if test.empty:
        raise ValueError(f"{path} has no rows with split == 'test'")
    return test


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_dummy_files(pred_dir: str | Path) -> set[str]:
    """Names of CSVs that are still the generated dummy data (hash listed in the manifest).
    A real file dropped in with the same name has a different hash, so it is not flagged."""
    pred_dir = Path(pred_dir)
    manifest = pred_dir / DUMMY_MANIFEST
    if not manifest.exists():
        return set()
    dummy_hashes = set(json.loads(manifest.read_text()).values())
    return {p.name for p in pred_dir.glob("*.csv") if file_sha256(p) in dummy_hashes}


# --------------------------------------------------------------------------- alignment check
@dataclass
class AlignmentResult:
    reference: str                 # what every file was compared against
    used_official_test: bool       # True once meld_clean.csv is in the repo
    n_rows: int                    # SHARED rows: the first n_rows test lines, present in every file
    frames: dict[str, pd.DataFrame]  # the usable files cut to the shared rows: score these, nothing else
    models: dict[str, str]         # filename -> model_name (usable files only)
    set_aside: dict[str, str]      # leftover DUMMY files that did not fit, with the reason
    notes: list[str]
    warnings: list[str]            # loud: something looks wrong even though rows line up
    full_frames: dict[str, pd.DataFrame] = field(default_factory=dict)  # usable files, all their rows
    test_rows: int | None = None   # size of the official test split (None without meld_clean.csv)

    @property
    def row_counts(self) -> dict[str, int]:
        return {name: len(df) for name, df in self.full_frames.items()}

    @property
    def longer_files(self) -> dict[str, pd.DataFrame]:
        """Files with more rows than the shared ones (e.g. a full-test-set BERT run next to
        100-line LLM runs). They also get scored on all their rows, separately."""
        return {n: df for n, df in self.full_frames.items() if len(df) > self.n_rows}


def _snippet(value: str, width: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 3] + "..."


def _fingerprint(df: pd.DataFrame) -> str:
    joined = "\x1f".join(df["utterance"]) + "\x1e" + "\x1f".join(df["true_label"])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _structure_problem(df: pd.DataFrame) -> str | None:
    """Columns, model_name and label checks for one file. Returns a message or None."""
    if list(df.columns) != COLUMNS:
        return (f"Columns are {list(df.columns)}\n"
                f"    They must be exactly {COLUMNS}, in this order.")
    if df.empty:
        return "The file has a header but no rows."
    names = df["model_name"].unique().tolist()
    if len(names) != 1:
        return f"model_name must be one value for the whole file, found {names[:5]}"
    if names[0] not in MODEL_NAMES:
        return f"model_name {names[0]!r} is not one of the fixed names {MODEL_NAMES}"
    for col in ("true_label", "predicted_label"):
        bad = np.flatnonzero(~df[col].isin(LABELS).to_numpy())
        if bad.size:
            i = int(bad[0])
            value = df[col].iloc[i]
            hint = ""
            if value.strip().lower() in LABELS:
                hint = "  (casing or whitespace problem: labels must be lowercase, no spaces)"
            elif value == "":
                hint = "  (empty: did an LLM response fail to parse?)"
            return (f"Invalid {col} at row {i} (line {i + 2} of the CSV): {value!r}{hint}\n"
                    f"    {bad.size} invalid {col} value(s) in total. Allowed: {LABELS}")
    return None


def allowed_rows(model: str, full: int) -> set[int]:
    """Row counts a file may have: the full test split for every model; the three LLM runs may
    instead be exactly the first LLM_RUN_ROWS test lines."""
    rows = {full}
    if model in LLM_MODELS and LLM_RUN_ROWS < full:
        rows.add(LLM_RUN_ROWS)
    return rows


def _alignment_problem(df: pd.DataFrame, ref: pd.DataFrame, model: str) -> str | None:
    """Every model: all reference rows, in the reference order. The three LLM runs may instead be
    exactly the first LLM_RUN_ROWS rows, in order, as save_predictions(..., full_test_set=False)
    writes them. Never shuffled, skipped, re-split or padded."""
    n, full = len(df), len(ref)
    if n not in allowed_rows(model, full):
        if model in LLM_MODELS:
            need = f"exactly the first {LLM_RUN_ROWS} test lines, or all {full}"
        else:
            need = f"all {full} test lines (only the LLM runs {', '.join(LLM_MODELS)} may be the first {LLM_RUN_ROWS})"
        if n > full:
            hint = "Extra rows mean train/dev lines or duplicates got in. Re-save with save_predictions()."
        else:
            hint = ("A wrong-sized file usually means a different live run (e.g. the 40-line prompt demo)\n"
                    "    overwrote the committed CSV. Restore it:  git checkout -- outputs/predictions/<file>")
        return f"Has {n} rows; {model} must have {need}.\n    {hint}"
    head = ref.iloc[:n]
    for col in ("utterance", "true_label"):
        mismatched = np.flatnonzero(df[col].to_numpy() != head[col].to_numpy())
        if mismatched.size:
            i = int(mismatched[0])
            capped = (f"\n    The LLM runs must be the FIRST {n} test lines, in order (save_predictions does this)."
                      if n < full else "")
            return (f"First mismatched row: {i} (line {i + 2} of the CSV), column '{col}'\n"
                    f"    expected: {_snippet(head[col].iloc[i])}\n"
                    f"    found:    {_snippet(df[col].iloc[i])}\n"
                    f"    {mismatched.size} of {n} rows differ. Rows were shuffled, dropped or re-split.{capped}")
    return None


def _sanity_warnings(name: str, df: pd.DataFrame) -> list[str]:
    """Rows can line up while predicted_label is in the wrong order (shuffled DataLoader,
    LLM batches collected out of order). Chance-level agreement gives that away."""
    acc = float((df["true_label"] == df["predicted_label"]).mean())
    if acc > 0.98:
        return [f"{name}: accuracy is {pct(acc)}, which is too good to be true. Was y_test passed as y_pred?"]
    kappa = cohen_kappa_score(df["true_label"], df["predicted_label"])
    if kappa < 0.05:
        return [f"{name}: predictions look unrelated to the rows (Cohen's kappa {kappa:.2f}, about chance).\n"
                "      Check that y_pred is in the same order as X_test: no shuffled DataLoader,\n"
                "      no LLM batches collected out of order. (A model that only ever predicts one class\n"
                "      also triggers this.)"]
    return []


def check_alignment(frames: dict[str, pd.DataFrame],
                    official_test: pd.DataFrame | None = None,
                    dummy_files: set[str] | frozenset = frozenset()) -> AlignmentResult:
    """Run this FIRST. Raises AlignmentError naming every bad REAL file and its first bad row.

    Reference: the official MELD test split if data/processed/meld_clean.csv exists, otherwise
    the version of (utterances, true labels) that a majority of the real files agree on.
    Leftover DUMMY files never vote, and a dummy that doesn't fit is set aside instead of
    stopping the run, so real files and dummies can share the folder while files trickle in.
    """
    if not frames:
        raise AlignmentError("No predictions CSVs to check.")

    problems: dict[str, str] = {}
    set_aside: dict[str, str] = {}
    notes: list[str] = []
    models: dict[str, str] = {}

    def reject(name: str, problem: str) -> None:
        if name in dummy_files:
            set_aside[name] = problem.splitlines()[0]
        else:
            problems[name] = problem
        models.pop(name, None)

    for name, df in frames.items():
        problem = _structure_problem(df)
        if problem:
            reject(name, problem)
            continue
        model = df["model_name"].iloc[0]
        clash = [other for other, m in models.items() if m == model]
        if clash:
            reject(name, f"model_name {model!r} is already used by {clash[0]}. Each model needs exactly one CSV.")
            continue
        models[name] = model
        if name != f"{model}_predictions.csv":
            notes.append(f"{name} holds model_name {model!r}; expected filename {model}_predictions.csv")

    ref, reference = None, ""
    if official_test is not None:
        ref, reference = official_test, "the official MELD test split (data/processed/meld_clean.csv)"
    elif models:
        real = [n for n in models if n not in dummy_files]
        voters = real or list(models)  # dummies only vote when no real file is present
        # Files can be capped runs of different lengths, so they vote on the rows they all share.
        shared = min(len(frames[n]) for n in voters)
        prints = {n: _fingerprint(frames[n].iloc[:shared]) for n in voters}
        winner, count = Counter(prints.values()).most_common(1)[0]
        if len(voters) > 1 and count * 2 <= len(voters):
            for n in voters:
                reject(n, "These files disagree with each other and no version has a majority, so there is\n"
                          "    no way to tell which one is right. Commit data/processed/meld_clean.csv so every\n"
                          "    file is checked against the official test split.")
        else:
            # The longest agreeing file is the reference, so shorter capped runs are checked as its prefix
            ref_name = max((n for n in voters if prints[n] == winner), key=lambda n: len(frames[n]))
            ref = frames[ref_name][["utterance", "true_label"]]
            kind = "real " if real else ""
            reference = f"{ref_name} (agreed by {count} of {len(voters)} {kind}file(s))"
            if len(voters) == 1:
                notes.append("Only one file to compare against: alignment can't really be verified "
                             "until meld_clean.csv is committed.")
        notes.append("data/processed/meld_clean.csv not found: CSVs were checked against each other, "
                     "not against the official test split.")

    if ref is not None:
        for name in list(models):
            problem = _alignment_problem(frames[name], ref, models[name])
            if problem:
                reject(name, problem)

    if problems:
        body = [f"Compared against: {reference or 'nothing usable'}",
                "Nothing has been scored. A chart built on misaligned rows looks clean and means nothing.", ""]
        for k, (name, problem) in enumerate(sorted(problems.items()), 1):
            body += [f"[{k}] {name}", f"    {problem}", ""]
        if set_aside:
            body += [f"(Also set aside {len(set_aside)} leftover DUMMY file(s): {', '.join(sorted(set_aside))})", ""]
        body += ["Fix: re-save with save_predictions() from the base notebook, on the official test set,",
                 "in the official order, never shuffled or re-split."]
        raise AlignmentError(_banner(
            f"ALIGNMENT CHECK FAILED: {len(problems)} of {len(frames)} predictions CSV(s) are not usable.", body))

    if not models:
        raise AlignmentError(_banner("NO USABLE PREDICTIONS FILES", [
            f"Set aside: {name}: {why}" for name, why in sorted(set_aside.items())] + [
            "Rerun make_dummy_predictions.py, or add real CSVs saved with save_predictions()."]))

    full = {n: frames[n] for n in models}
    # Every model is scored on the rows ALL files have: the first n_shared test lines, in order.
    n_shared = min(len(df) for df in full.values())
    usable = {n: df.iloc[:n_shared].reset_index(drop=True) for n, df in full.items()}
    if len({len(df) for df in full.values()}) > 1:
        counts = ", ".join(f"{models[n]} {len(df):,}" for n, df in sorted(full.items(), key=lambda x: len(x[1])))
        notes.append(f"Files cover different numbers of test lines: {counts}.\n"
                     f"        Every model is scored on the shared first {n_shared:,} lines so the comparison is fair.\n"
                     "        Files with more rows are also scored on all of them, in a separate table.")
    warnings = [w for name, df in full.items() for w in _sanity_warnings(name, df)]
    return AlignmentResult(reference=reference, used_official_test=official_test is not None,
                           n_rows=n_shared, frames=usable, models=dict(models), set_aside=set_aside,
                           notes=notes, warnings=warnings, full_frames=full,
                           test_rows=len(official_test) if official_test is not None else None)


def describe_rows(result: AlignmentResult) -> str:
    """'the first 100 of 2,610 official test lines' / 'all 2,610 official test lines'."""
    n, total = result.n_rows, result.test_rows
    if total is None:
        return f"{n:,} test lines (no meld_clean.csv: checked against each other)"
    return f"all {total:,} official test lines" if n == total else f"the first {n:,} of {total:,} official test lines"


def print_alignment(result: AlignmentResult) -> None:
    print(f"ALIGNMENT CHECK PASSED: {len(result.models)} file(s). utterance and true_label match the "
          "reference, row for row.")
    print(f"Reference: {result.reference}")
    print(f"Scored on: {describe_rows(result)} (the rows every file has)")
    counts = result.row_counts
    for name, model in result.models.items():
        rows = f"{counts[name]:,} rows" if name in counts else ""
        print(f"  OK         {name:<36} model_name = {model:<18} {rows}")
    for name, why in result.set_aside.items():
        print(f"  SET ASIDE  {name:<36} leftover DUMMY file that doesn't fit: {why}")
        print(f"             (delete it, or rerun make_dummy_predictions.py)")
    for n in result.notes:
        print(f"  NOTE  {n}")
    for w in result.warnings:
        print(f"\n  WARNING  {w}")


# --------------------------------------------------------------------------- scoring
@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    macro_f1: float
    report_text: str
    report_dict: dict
    confusion: np.ndarray  # rows = true label, columns = predicted label, order = LABELS
    n_rows: int
    is_dummy: bool = False
    y_true: np.ndarray = field(default=None, repr=False)
    y_pred: np.ndarray = field(default=None, repr=False)

    @property
    def title(self) -> str:
        name = display_name(self.model_name).replace("\n", " ")
        return f"{name} (DUMMY DATA)" if self.is_dummy else name


def score(model_name: str, y_true, y_pred, is_dummy: bool = False) -> ModelResult:
    y_true, y_pred = np.asarray(list(y_true), dtype=object), np.asarray(list(y_pred), dtype=object)
    return ModelResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        report_text=classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0),
        report_dict=classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True),
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS),
        n_rows=len(y_true),
        is_dummy=is_dummy,
        y_true=y_true,
        y_pred=y_pred,
    )


def evaluate_all(frames: dict[str, pd.DataFrame], dummy_files: set[str] = frozenset()) -> list[ModelResult]:
    """Score every (already aligned) file, in the fixed model order.
    Pass AlignmentResult.frames, which are already cut to the shared rows."""
    lengths = {len(df) for df in frames.values()}
    if len(lengths) > 1:
        raise ValueError(f"Frames have different lengths {sorted(lengths)}: pass AlignmentResult.frames "
                         "(cut to the shared rows), not the raw files.")
    results = [score(df["model_name"].iloc[0], df["true_label"], df["predicted_label"],
                     is_dummy=name in dummy_files)
               for name, df in frames.items()]
    order = {m: i for i, m in enumerate(MODEL_NAMES)}
    return sorted(results, key=lambda r: order.get(r.model_name, len(order)))


def evaluate_longer(alignment: AlignmentResult, dummy_files: set[str] = frozenset()) -> list[ModelResult]:
    """Models whose file covers more than the shared rows, scored on ALL their rows.
    Only files of the same length are compared with each other, so this returns the models
    that cover the most rows (normally the full test set: BERT and the BiLSTM)."""
    longer = alignment.longer_files
    if not longer:
        return []
    most = max(len(df) for df in longer.values())
    return evaluate_all({n: df for n, df in longer.items() if len(df) == most}, dummy_files)


def majority_baseline(y_true, majority: str = MAJORITY_LABEL) -> ModelResult:
    """A 'model' that ignores the text and always predicts the majority class from training."""
    return score(f"always_{majority}", list(y_true), [majority] * len(y_true))


def top_confusions(result: ModelResult, k: int = 3, include_neutral: bool = True,
                   min_count: int = 2) -> list[tuple[str, str, float, int]]:
    """Most common mistakes as (true, predicted, share of that true class, count).
    Mistakes made on fewer than min_count lines are skipped when anything bigger exists: on 100 test
    lines there are only 4 fear lines, and one stray line (25%) is not a pattern."""
    cm = result.confusion
    rows = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    pairs = []
    for i, t in enumerate(LABELS):
        for j, p in enumerate(LABELS):
            if i == j or (not include_neutral and "neutral" in (t, p)):
                continue
            pairs.append((t, p, float(rate[i, j]), int(cm[i, j])))
    pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
    big = [x for x in pairs if x[3] >= min_count]
    return (big + [x for x in pairs if x[3] < min_count])[:k]


def lines(n: int) -> str:
    return f"{n} line" if n == 1 else f"{n} lines"


# --------------------------------------------------------------------------- is #1 really ahead?
@dataclass
class Gap:
    leader: str
    runner_up: str
    diff: float  # macro F1 points, leader minus runner-up (0-1 scale)
    low: float   # 95% paired bootstrap interval
    high: float

    @property
    def clear(self) -> bool:
        return self.low > 0


def _macro_f1_resampled(t: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Macro F1 for each bootstrap resample (rows of idx), vectorised. Matches sklearn's
    f1_score(labels=LABELS, average='macro', zero_division=0)."""
    k, n_boot = len(LABELS), idx.shape[0]
    codes = t[idx] * k + p[idx] + (np.arange(n_boot)[:, None] * k * k)
    cm = np.bincount(codes.ravel(), minlength=n_boot * k * k).reshape(n_boot, k, k)
    tp = cm[:, np.arange(k), np.arange(k)]
    denom = cm.sum(axis=1) + cm.sum(axis=2)  # predicted totals + true totals = 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(tp.shape, dtype=float), where=denom > 0)
    return f1.mean(axis=1)


def macro_f1_gap(leader: ModelResult, runner_up: ModelResult, n_boot: int = 1000, seed: int = 27) -> Gap:
    """Paired bootstrap: resample the same test rows for both models, 1,000 times."""
    lut = {label: i for i, label in enumerate(LABELS)}
    t = np.array([lut[x] for x in leader.y_true])
    pa = np.array([lut[x] for x in leader.y_pred])
    pb = np.array([lut[x] for x in runner_up.y_pred])
    idx = np.random.default_rng(seed).integers(0, len(t), size=(n_boot, len(t)))
    diffs = _macro_f1_resampled(t, pa, idx) - _macro_f1_resampled(t, pb, idx)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return Gap(leader.model_name, runner_up.model_name, leader.macro_f1 - runner_up.macro_f1,
               float(low), float(high))


def describe_gap(gap: Gap) -> str:
    verdict = ("The whole interval is above zero, so #1 really is ahead." if gap.clear else
               "The interval includes zero, so on this test set it is too close to call.")
    return (f"Is #1 really ahead? {gap.leader} leads {gap.runner_up} by {gap.diff * 100:+.1f} macro-F1 points "
            f"(95% bootstrap interval {gap.low * 100:+.1f} to {gap.high * 100:+.1f}).\n{verdict}")


# --------------------------------------------------------------------------- reports
FULL_SUFFIX = "_all_rows"  # outputs of a model scored on all its rows, beyond the shared ones


def save_report(result: ModelResult, reports_dir: str | Path, suffix: str = "") -> Path:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{result.model_name}_classification_report{suffix}.txt"
    header = [f"Model:     {result.model_name}{'   *** DUMMY DATA, NOT REAL RESULTS ***' if result.is_dummy else ''}",
              f"Rows:      {result.n_rows}",
              f"Accuracy:  {result.accuracy:.4f}",
              f"Macro F1:  {result.macro_f1:.4f}", ""]
    path.write_text("\n".join(header) + result.report_text)
    per_class = {k: v for k, v in result.report_dict.items() if k != "accuracy"}
    table = pd.DataFrame(per_class).T
    table.loc["accuracy"] = [np.nan, np.nan, result.accuracy, result.n_rows]
    table.to_csv(reports_dir / f"{result.model_name}_classification_report{suffix}.csv")
    return path


def remove_stale_outputs(reports_dir: str | Path, figures_dir: str | Path, present_models,
                         longer_models=()) -> list[str]:
    """Delete per-model reports/figures of models that are no longer in the predictions folder
    (and all-rows outputs of models that no longer have extra rows), so an old (possibly dummy)
    confusion matrix can't end up in the deck."""
    present, longer = set(present_models), set(longer_models)
    removed = []
    for m in MODEL_NAMES:
        stale = [] if m in present else [""]
        if m not in longer:
            stale.append(FULL_SUFFIX)
        for s in stale:
            for p in (Path(figures_dir) / f"{m}_confusion_matrix{s}.png",
                      Path(reports_dir) / f"{m}_classification_report{s}.txt",
                      Path(reports_dir) / f"{m}_classification_report{s}.csv"):
                if p.exists():
                    p.unlink()
                    removed.append(p.name)
    if not longer:
        p = Path(reports_dir) / f"scoreboard{FULL_SUFFIX}.csv"
        if p.exists():
            p.unlink()
            removed.append(p.name)
    return removed


def scoreboard(results: list[ModelResult]) -> pd.DataFrame:
    """Ranked by macro F1 (the honest metric), with the accuracy rank alongside to show where they disagree."""
    board = pd.DataFrame({
        "model": [r.model_name for r in results],
        "approach": [APPROACH.get(r.model_name, "") for r in results],
        "macro_f1": [r.macro_f1 for r in results],
        "accuracy": [r.accuracy for r in results],
        "rows": [r.n_rows for r in results],
        "dummy": [r.is_dummy for r in results],
    }).sort_values(["macro_f1", "accuracy"], ascending=False, ignore_index=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    board["accuracy_rank"] = board["accuracy"].rank(ascending=False, method="min").astype(int)
    return board


def format_scoreboard(board: pd.DataFrame, baseline: ModelResult | None = None) -> str:
    lines = [f"{'Rank':<5} {'Model':<20} {'Approach':<14} {'Macro F1':>9} {'Accuracy':>9} {'Acc. rank':>10}",
             "-" * 72]
    for row in board.itertuples():
        tag = "   (DUMMY)" if row.dummy else ""
        lines.append(f"{row.rank:<5} {row.model:<20} {row.approach:<14} {pct(row.macro_f1):>9} "
                     f"{pct(row.accuracy):>9} {row.accuracy_rank:>10}{tag}")
    if baseline is not None:
        lines += ["-" * 72,
                  f"{'-':<5} {baseline.model_name:<20} {'no model':<14} {pct(baseline.macro_f1):>9} "
                  f"{pct(baseline.accuracy):>9} {'':>10}   (baseline)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib.pyplot as plt
    return plt.rc_context({
        "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.edgecolor": GREY,
        "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY, "text.color": NAVY,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def plot_confusion_matrix(result: ModelResult, out_path: str | Path | None = None):
    """Row-normalised: each row shows where the lines of one TRUE emotion ended up."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = result.confusion
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape, dtype=float), where=rows > 0)
    annot = np.array([[f"{share[i, j] * 100:.0f}%\n{cm[i, j]}" for j in range(len(LABELS))]
                      for i in range(len(LABELS))])

    with _style():
        fig, ax = plt.subplots(figsize=(11, 9.5))
        sns.heatmap(share, annot=annot, fmt="", cmap=sns.light_palette(BLUE, as_cmap=True),
                    vmin=0, vmax=1, linewidths=1.5, linecolor="white", cbar=False, square=True,
                    xticklabels=LABELS, yticklabels=LABELS, annot_kws={"fontsize": 15}, ax=ax)
        ax.set_xlabel("What the model predicted", labelpad=12)
        ax.set_ylabel("What the line really was", labelpad=12)
        ax.tick_params(axis="both", length=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_title(f"{result.title}: where it gets confused", loc="left", pad=40, fontweight="bold",
                     color=WARN_RED if result.is_dummy else NAVY)
        ax.text(0, 1.02, f"{result.n_rows:,} test lines. Each row: where did the lines of that true emotion go? "
                "Diagonal = correct.", transform=ax.transAxes, fontsize=13, color=GREY)
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_master_comparison(results: list[ModelResult], baseline: ModelResult | None = None,
                           out_path: str | Path | None = None, n_rows: int | None = None,
                           used_official_test: bool = True, test_rows: int | None = None):
    """The climax chart: every model, accuracy and macro F1 side by side, ranked by macro F1.
    The leader is only crowned if a paired bootstrap says the gap to #2 is bigger than noise."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ranked = sorted(results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True)
    any_dummy = any(r.is_dummy for r in ranked)
    gap = macro_f1_gap(ranked[0], ranked[1]) if len(ranked) >= 2 else None
    width = 0.36

    with _style():
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2)

        xs = list(np.arange(len(ranked), dtype=float))
        groups = [(x, r, False) for x, r in zip(xs, ranked)]
        if baseline is not None:
            groups.append((len(ranked) + 0.35, baseline, True))

        top = max(max(r.accuracy, r.macro_f1) for _, r, _ in groups)
        ymax = min(100, top * 100 + 22)

        for x, r, is_base in groups:
            acc_kw = dict(color="#D5D8DD", edgecolor=GREY, hatch="//", linewidth=1.2) if is_base \
                else dict(color=LIGHT_BLUE, edgecolor="none")
            f1_kw = dict(color="#8E9399", edgecolor=GREY, hatch="//", linewidth=1.2) if is_base \
                else dict(color=BLUE, edgecolor="none")
            for offset, value, kw, weight in ((-width / 2, r.accuracy, acc_kw, "normal"),
                                              (width / 2, r.macro_f1, f1_kw, "bold")):
                ax.bar(x + offset, value * 100, width, zorder=3, **kw)
                ax.text(x + offset, value * 100 + 1.2, f"{value * 100:.1f}%", ha="center", va="bottom",
                        fontsize=18, fontweight=weight, color=GREY if is_base else NAVY, zorder=4)

        # Crown the leader only when the gap is real; otherwise say it's a statistical tie
        if gap is not None and gap.clear:
            ax.text(0, max(ranked[0].accuracy, ranked[0].macro_f1) * 100 + 8, "★ WINNER",
                    ha="center", va="bottom", fontsize=20, fontweight="bold", color=BLUE)
        elif gap is not None:
            y = max(ranked[0].accuracy, ranked[0].macro_f1, ranked[1].accuracy, ranked[1].macro_f1) * 100 + 8
            ax.annotate("", xy=(-0.35, y), xytext=(1.35, y),
                        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=2.5), zorder=4)
            ax.text(0.5, y + 1.2, "TOO CLOSE TO CALL", ha="center", va="bottom",
                    fontsize=20, fontweight="bold", color=BLUE)

        # Baseline separator and callout
        if baseline is not None:
            bx = groups[-1][0]
            ax.axvline(bx - 0.62, color=GREY, linestyle=":", linewidth=1.5, zorder=1)
            ax.text(bx, max(baseline.accuracy, baseline.macro_f1) * 100 + 8,
                    "Never reads the text:\ngood accuracy,\nuseless model",
                    ha="center", va="bottom", fontsize=15, color=GREY, style="italic", linespacing=1.3)

        # X axis: model name, then approach underneath
        labels = [display_name(r.model_name) if not is_base else f"Always says\n'{r.model_name.split('_', 1)[1]}'"
                  for _, r, is_base in groups]
        ax.set_xticks([x for x, _, _ in groups])
        ax.set_xticklabels(labels, fontsize=19, fontweight="bold")
        for x, r, is_base in groups:
            tag = "baseline" if is_base else APPROACH.get(r.model_name, "")
            ax.text(x, -0.155, tag, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=15, color=WARN_RED if r.is_dummy else GREY, style="italic")
        ax.tick_params(axis="x", length=0, pad=10)

        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.6, groups[-1][0] + 0.6)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.grid(axis="y", color="#E6E8EC", linewidth=1, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Title block
        fig.text(0.07, 0.935, "The scoreboard: who actually won?", fontsize=34, fontweight="bold", color=NAVY)
        rows_txt = f"{n_rows:,} " if n_rows else ""
        if used_official_test and n_rows and test_rows and n_rows < test_rows:
            where = f"Same first {n_rows:,} lines of MELD's official test split"
        elif used_official_test:
            where = f"Same {rows_txt}lines of MELD's official test split"
        else:
            where = f"Same {rows_txt}test lines for every model"
        fig.text(0.07, 0.885, f"{where} · ranked by macro F1 (all 7 emotions count equally)",
                 fontsize=17, color=GREY)
        handles = [Patch(color=LIGHT_BLUE, label="Accuracy: share of lines labelled right"),
                   Patch(color=BLUE, label="Macro F1: average score across all 7 emotions")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.865), ncol=2,
                   frameon=False, fontsize=16, handlelength=1.4, columnspacing=2.5)

        fig.text(0.07, 0.03, "AI Ignite at York · Forge Workshop 01 · Emotion detection on MELD",
                 fontsize=12, color=GREY)
        if any_dummy:  # drawn on the axes behind the bars (zorder 3) so it never tints them
            ax.text(0.5, 0.5, "DUMMY DATA", rotation=18, fontsize=130, color=WARN_RED, alpha=0.08,
                    ha="center", va="center", fontweight="bold", zorder=0.5, transform=ax.transAxes)
            fig.text(0.98, 0.03, "Contains DUMMY predictions: not real results", ha="right",
                     fontsize=14, color=WARN_RED, fontweight="bold")

        if out_path:
            fig.savefig(out_path, dpi=200)
    return fig


# --------------------------------------------------------------------------- one-shot runner
def run_pipeline(repo_root: str | Path | None = None, pred_dir: str | Path | None = None,
                 verbose: bool = True, out_dir: str | Path | None = None) -> dict:
    """Everything the notebook does, in one call (used by the CLI and tests).
    out_dir: where reports/ and figures/ go (default: the repo's outputs/)."""
    import matplotlib.pyplot as plt

    root = find_repo_root(repo_root)
    pred_dir = Path(pred_dir or os.environ.get("FORGE_PREDICTIONS_DIR") or root / "outputs" / "predictions")
    out_dir = Path(out_dir) if out_dir else root / "outputs"
    reports_dir, figures_dir = out_dir / "reports", out_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames = load_predictions(pred_dir)
    dummies = find_dummy_files(pred_dir)
    alignment = check_alignment(frames, load_official_test(root), dummies)
    results = evaluate_all(alignment.frames, dummies)
    y_true = next(iter(alignment.frames.values()))["true_label"].tolist()
    baseline = majority_baseline(y_true)
    longer = evaluate_longer(alignment, dummies)

    remove_stale_outputs(reports_dir, figures_dir, [r.model_name for r in results],
                         [r.model_name for r in longer])
    for r in results:
        save_report(r, reports_dir)
        plt.close(plot_confusion_matrix(r, figures_dir / f"{r.model_name}_confusion_matrix.png"))
    for r in longer:
        save_report(r, reports_dir, FULL_SUFFIX)
        plt.close(plot_confusion_matrix(r, figures_dir / f"{r.model_name}_confusion_matrix{FULL_SUFFIX}.png"))
    board = scoreboard(results)
    board.to_csv(reports_dir / "scoreboard.csv", index=False)
    longer_board = scoreboard(longer) if longer else None
    if longer_board is not None:
        longer_board.to_csv(reports_dir / f"scoreboard{FULL_SUFFIX}.csv", index=False)
    plt.close(plot_master_comparison(results, baseline, figures_dir / "master_comparison.png",
                                     n_rows=alignment.n_rows, used_official_test=alignment.used_official_test,
                                     test_rows=alignment.test_rows))
    if verbose:
        print_alignment(alignment)
        print(f"\nTHE SCOREBOARD, on {describe_rows(alignment)}")
        print(format_scoreboard(board, baseline))
        if len(results) >= 2:
            ranked = sorted(results, key=lambda r: (r.macro_f1, r.accuracy), reverse=True)
            print("\n" + describe_gap(macro_f1_gap(ranked[0], ranked[1])))
        if longer:
            n = longer[0].n_rows
            print(f"\nALSO SCORED ON ALL {n:,} ROWS (only models whose file has them; not comparable with the table above)")
            print(format_scoreboard(longer_board, majority_baseline(longer[0].y_true)))
    return {"alignment": alignment, "results": results, "baseline": baseline, "board": board,
            "longer": longer, "longer_board": longer_board}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Forge Workshop 01 evaluation pipeline.")
    parser.add_argument("--pred-dir", help="predictions folder (default: outputs/predictions)")
    args = parser.parse_args()
    run_pipeline(pred_dir=args.pred_dir)
