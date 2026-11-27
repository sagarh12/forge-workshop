"""
Tests for the alignment check and scoring in notebooks/sheheryar/eval.py.

    python tests/test_alignment.py        (no extra packages needed)
    pytest tests/                          (also works)

Two kinds of test:
  * The COMMITTED files in outputs/predictions/ (the real predictions) must pass the alignment
    check against the official MELD test split and run through the whole pipeline. Run this
    before committing a predictions CSV.
  * Everything else runs on dummy predictions generated in a temp folder (all five models, the
    full test set, built on the real official test rows once data/processed/meld_clean.csv is in
    the repo). Nothing is ever written into outputs/.

Every predictions file covers the whole test split, except the three LLM runs (zero_shot, few_shot,
balanced_few_shot), which may be exactly the first 100 test lines (save_predictions(...,
full_test_set=False)). Every model is then scored on the lines all files share.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "sheheryar"))
import eval as ev  # noqa: E402

COMMITTED_DIR = ROOT / "outputs" / "predictions"
FIXTURE = ROOT / "tests" / "fixtures" / "misaligned_bert_finetuned_predictions.csv"
LLM_MODELS = ("zero_shot", "few_shot", "balanced_few_shot")


def _make_dummies() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="forge-test-dummies-"))
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "sheheryar" / "make_dummy_predictions.py"),
                    "--pred-dir", str(tmp / "predictions"), "--fixtures-dir", str(tmp / "fixtures")],
                   check=True, capture_output=True)
    return tmp / "predictions"


DUMMY_DIR = _make_dummies()


def fresh() -> dict[str, pd.DataFrame]:
    """All five models, full test set (dummy predictions)."""
    return ev.load_predictions(DUMMY_DIR)


def capped(frames: dict, n: int = 100, models=LLM_MODELS) -> dict:
    """Cut the LLM files to their first n rows, the way a capped save_predictions() run is saved."""
    for m in models:
        frames[f"{m}_predictions.csv"] = frames[f"{m}_predictions.csv"].head(n).copy()
    return frames


def official():
    return ev.load_official_test(ROOT)


_OFFICIAL = object()  # default: check against the repo's official test split (None = without it)


def expect_failure(frames, *must_contain: str, official_test=_OFFICIAL, dummy_files=frozenset()) -> str:
    try:
        ev.check_alignment(frames, official() if official_test is _OFFICIAL else official_test, dummy_files)
    except ev.AlignmentError as err:
        msg = str(err)
        for text in must_contain:
            assert text in msg, f"expected {text!r} in the error, got:\n{msg}"
        return msg
    raise AssertionError("alignment check PASSED but should have FAILED")


def swap_rows(df: pd.DataFrame, start: int = 1233) -> int:
    """Swap two neighbouring rows whose utterances differ; return the first index."""
    i = next(j for j in range(start, len(df) - 1) if df["utterance"].iloc[j] != df["utterance"].iloc[j + 1])
    df.iloc[[i, i + 1]] = df.iloc[[i + 1, i]].to_numpy()
    return i


# ------------------------------------------------------------------ the committed (real) files
def test_committed_files_pass_against_the_official_split():
    if not any(COMMITTED_DIR.glob("*.csv")):
        print("    (no committed predictions yet: nothing to check)")
        return
    frames = ev.load_predictions(COMMITTED_DIR)
    dummies = ev.find_dummy_files(COMMITTED_DIR)
    result = ev.check_alignment(frames, official(), dummies)
    real = set(frames) - dummies
    assert real <= set(result.models), f"real files not usable: {real - set(result.models)}"
    assert not result.warnings, result.warnings
    if official() is not None:
        assert result.used_official_test
        assert result.n_rows in (ev.LLM_RUN_ROWS, len(official()))


def test_committed_files_run_through_the_whole_pipeline():
    if not any(COMMITTED_DIR.glob("*.csv")):
        return
    with tempfile.TemporaryDirectory() as out:
        run = ev.run_pipeline(ROOT, COMMITTED_DIR, verbose=False, out_dir=out)
        assert (Path(out) / "figures" / "master_comparison.png").exists()
        assert (Path(out) / "reports" / "scoreboard.csv").exists()
        assert len(run["board"]) == len(run["alignment"].models)


# ------------------------------------------------------------------ full-length files
def test_good_files_pass():
    frames = fresh()
    result = ev.check_alignment(frames, official())
    assert set(result.models) == set(frames)
    assert result.n_rows == len(next(iter(frames.values())))
    assert not result.longer_files
    assert not result.warnings, result.warnings


def test_misaligned_fixture_fails():
    frames = fresh()
    frames["bert_finetuned_predictions.csv"] = ev.read_predictions_csv(FIXTURE)
    expected = ["ALIGNMENT CHECK FAILED", "bert_finetuned_predictions.csv"]
    if official() is not None:  # built on the real test rows: two neighbouring lines swapped
        expected.append("First mismatched row: 1233")
    expect_failure(frames, *expected)


def test_two_swapped_rows_fail_at_the_right_row():
    frames = fresh()
    name = next(iter(frames))
    i = swap_rows(frames[name])
    expect_failure(frames, name, f"First mismatched row: {i}")


def test_uppercase_label_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name].loc[10, "predicted_label"] = "Anger"
    expect_failure(frames, name, "row 10", "lowercase")


def test_wrong_column_order_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name] = frames[name][["true_label", "utterance", "predicted_label", "model_name"]]
    expect_failure(frames, name, "Columns are")


def test_unknown_model_name_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name]["model_name"] = "bert"
    expect_failure(frames, name, "not one of the fixed names")


def test_duplicate_model_fails():
    frames = fresh()
    name, df = next(iter(frames.items()))
    frames["copy_of_" + name] = df.copy()
    expect_failure(frames, "already used by")


def test_missing_model_still_scores():
    frames = fresh()
    frames.pop(next(iter(frames)))
    result = ev.check_alignment(frames, official())
    assert len(ev.evaluate_all(result.frames)) == len(frames)


def test_shuffled_predictions_are_flagged():
    frames = fresh()
    name = next(iter(frames))
    df = frames[name]
    df["predicted_label"] = df["predicted_label"].sample(frac=1, random_state=0).to_numpy()
    result = ev.check_alignment(frames, official())  # rows still line up, so it passes...
    assert any(name in w and "unrelated" in w for w in result.warnings), result.warnings  # ...but warns


def test_file_longer_than_the_test_set_fails():
    frames = fresh()
    name = "bilstm_predictions.csv"
    frames[name] = pd.concat([frames[name], frames[name].head(5)], ignore_index=True)
    expect_failure(frames, name, "Extra rows", "bilstm must have all")


# ------------------------------------------------------------------ capped runs (first N test lines)
def test_capped_runs_pass_and_every_model_is_scored_on_the_shared_rows():
    full = fresh()
    frames = capped(fresh())
    result = ev.check_alignment(frames, official())
    total = len(full["bert_finetuned_predictions.csv"])
    assert result.n_rows == 100
    assert result.test_rows == (total if official() is not None else None)
    assert all(len(df) == 100 for df in result.frames.values())
    assert result.row_counts["bert_finetuned_predictions.csv"] == total
    assert set(result.longer_files) == {"bert_finetuned_predictions.csv", "bilstm_predictions.csv"}

    # BERT on the scoreboard = BERT scored on exactly the first 100 test lines
    results = {r.model_name: r for r in ev.evaluate_all(result.frames)}
    bert = full["bert_finetuned_predictions.csv"].head(100)
    assert results["bert_finetuned"].n_rows == 100
    assert abs(results["bert_finetuned"].macro_f1 -
               ev.score("bert_finetuned", bert["true_label"], bert["predicted_label"]).macro_f1) < 1e-12

    # ...and the trained models also get their full-test-set score, separately
    longer = ev.evaluate_longer(result)
    assert sorted(r.model_name for r in longer) == ["bert_finetuned", "bilstm"]
    assert all(r.n_rows == total for r in longer)


def test_capped_run_that_is_not_the_first_rows_fails():
    frames = fresh()
    name = "zero_shot_predictions.csv"
    frames[name] = frames[name].iloc[100:200].reset_index(drop=True)
    expect_failure(frames, name, "First mismatched row: 0", "FIRST 100 test lines")


def test_shuffled_capped_run_fails():
    frames = capped(fresh())
    name = "few_shot_predictions.csv"
    i = swap_rows(frames[name], start=40)
    expect_failure(frames, name, f"First mismatched row: {i}")


def test_capped_run_shorter_than_100_lines_fails():
    frames = fresh()
    name = "balanced_few_shot_predictions.csv"
    frames[name] = frames[name].head(40)
    expect_failure(frames, name, "Has 40 rows", "exactly the first 100")


def test_trained_model_with_100_rows_fails():
    """Only the three LLM runs may be the first 100 lines; BERT and the BiLSTM need all of them."""
    frames = fresh()
    total = len(frames["bert_finetuned_predictions.csv"])
    frames["bert_finetuned_predictions.csv"] = frames["bert_finetuned_predictions.csv"].head(100)
    expect_failure(frames, "bert_finetuned_predictions.csv", "Has 100 rows", f"bert_finetuned must have all {total}")


def test_llm_run_must_be_exactly_100_rows():
    """150 or 99 rows: not the agreed 100-line run, not the full test set."""
    for n in (150, 99):
        frames = fresh()
        frames["zero_shot_predictions.csv"] = frames["zero_shot_predictions.csv"].head(n)
        expect_failure(frames, "zero_shot_predictions.csv", f"Has {n} rows", "exactly the first 100")


def test_allowed_rows():
    assert ev.allowed_rows("few_shot", 2610) == {100, 2610}
    assert ev.allowed_rows("bilstm", 2610) == {2610}
    assert ev.allowed_rows("bert_finetuned", 2610) == {2610}


def test_evaluate_all_refuses_files_of_different_lengths():
    try:
        ev.evaluate_all(capped(fresh()))
    except ValueError as err:
        assert "AlignmentResult.frames" in str(err)
        return
    raise AssertionError("scoring files of different lengths side by side must be refused")


def test_capped_runs_vote_on_shared_rows_without_meld_clean():
    frames = capped(fresh())
    result = ev.check_alignment(frames, None)
    assert result.n_rows == 100 and len(result.models) == 5
    assert "agreed by 5 of 5" in result.reference, result.reference
    # The longest agreeing file is the reference, so the capped runs were checked as its prefix
    assert result.reference.split()[0] in {"bert_finetuned_predictions.csv", "bilstm_predictions.csv"}


# ------------------------------------------------------------------ real files mixed with dummies
def _as_real(frames: dict, names: list[str]) -> None:
    """Turn some dummy frames into stand-ins for real files: different (shared) utterances."""
    for n in names:
        frames[n]["utterance"] = [f"real line {i}" for i in range(len(frames[n]))]


def test_leftover_dummies_never_outvote_real_files():
    frames = fresh()
    names = sorted(frames)
    real, dummies = names[:2], set(names[2:])
    _as_real(frames, real)
    result = ev.check_alignment(frames, None, dummies)
    assert set(result.models) == set(real), result.models
    assert set(result.set_aside) == dummies, result.set_aside


def test_misaligned_real_file_still_fails_next_to_dummies():
    frames = fresh()
    names = sorted(frames)
    real, dummies = names[:3], set(names[3:])
    _as_real(frames, real)
    swap_rows(frames[real[0]])
    expect_failure(frames, real[0], official_test=None, dummy_files=dummies)


def test_no_majority_is_reported_not_guessed():
    frames = fresh()
    a, b = sorted(frames)[:2]
    two = {a: frames[a], b: frames[b]}
    swap_rows(two[b])
    expect_failure(two, "no version has a majority", official_test=None)


# ------------------------------------------------------------------ the path that runs on the day
TRICKY = ["Hello, world", 'She said "hi"', "line one\nline two", "  leading spaces", "trailing  ",
          "1", "café ’quote’", " nbsp", "Okay.", "Okay.", "NA", "None"]


def test_official_split_round_trip():
    """A tiny meld_clean.csv, read and saved exactly the way the base notebook does it
    (Step 4, Step 6, save_predictions), including a capped 100-line LLM run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data" / "processed").mkdir(parents=True)
        (tmp / "preds").mkdir()
        rng = np.random.default_rng(0)
        n_train, n_dev, n_test = 72, 48, 240
        n = n_train + n_dev + n_test
        utterances = [TRICKY[i % len(TRICKY)] + f" #{i}" if i % 3 else TRICKY[i % len(TRICKY)] for i in range(n)]
        meld = pd.DataFrame({
            "utterance": utterances,
            "label": rng.choice(ev.LABELS, n),
            "dialogue_id": [i // 6 for i in range(n)],
            "utterance_id": [i % 6 for i in range(n)],
            "speaker": "A",
            "split": ["train"] * n_train + ["dev"] * n_dev + ["test"] * n_test,
        })
        meld.to_csv(tmp / "data" / "processed" / "meld_clean.csv", index=False)

        base = pd.read_csv(tmp / "data" / "processed" / "meld_clean.csv")
        test_df = base[base["split"] == "test"].reset_index(drop=True)
        X_test, y_test = test_df["utterance"].tolist(), test_df["label"].tolist()
        for model, rows in (("bert_finetuned", n_test), ("zero_shot", 100)):
            pd.DataFrame({"utterance": X_test[:rows], "true_label": y_test[:rows],
                          "predicted_label": rng.choice(ev.LABELS, rows), "model_name": model}
                         ).to_csv(tmp / "preds" / f"{model}_predictions.csv", index=False)

        official_tmp = ev.load_official_test(tmp)
        frames = ev.load_predictions(tmp / "preds")
        result = ev.check_alignment(frames, official_tmp)
        assert result.used_official_test and result.test_rows == n_test
        assert result.n_rows == 100 and len(result.models) == 2
        assert set(result.longer_files) == {"bert_finetuned_predictions.csv"}

        i = swap_rows(frames["zero_shot_predictions.csv"], start=3)
        expect_failure(frames, "zero_shot_predictions.csv", f"First mismatched row: {i}",
                       official_test=official_tmp)


def test_unreadable_file_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bilstm_predictions.csv"
        bad.write_bytes(b"utterance,true_label,predicted_label,model_name\n\x92oops,anger,anger,bilstm\n")
        try:
            ev.load_predictions(tmp)
        except ev.AlignmentError as err:
            assert "bilstm_predictions.csv" in str(err)
            return
        raise AssertionError("an unreadable file should raise AlignmentError naming it")


def test_pipeline_with_capped_and_full_files_end_to_end():
    """Trained models on all test lines + 100-line LLM runs: scoreboard on the shared 100 lines,
    full-test-set table alongside, and no stale all-rows outputs once the long files are gone."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        preds, out = tmp / "preds", tmp / "out"
        preds.mkdir()
        for name, df in capped(fresh()).items():
            df.to_csv(preds / name, index=False)
        run = ev.run_pipeline(ROOT, preds, verbose=False, out_dir=out)
        assert set(run["board"]["rows"]) == {100} and len(run["board"]) == 5
        assert sorted(run["longer_board"]["model"]) == ["bert_finetuned", "bilstm"]
        for f in ("figures/master_comparison.png", "figures/bert_finetuned_confusion_matrix_all_rows.png",
                  "reports/scoreboard_all_rows.csv", "reports/bilstm_classification_report_all_rows.txt"):
            assert (out / f).exists(), f

        (preds / "bert_finetuned_predictions.csv").unlink()
        (preds / "bilstm_predictions.csv").unlink()
        ev.run_pipeline(ROOT, preds, verbose=False, out_dir=out)
        leftovers = sorted(p.name for p in out.rglob("*") if "bert" in p.name or "bilstm" in p.name
                           or "all_rows" in p.name)
        assert not leftovers, leftovers


# ------------------------------------------------------------------ metrics
def test_bootstrap_macro_f1_matches_sklearn():
    frames = fresh()
    r = ev.evaluate_all(frames)[0]
    lut = {label: i for i, label in enumerate(ev.LABELS)}
    t = np.array([lut[x] for x in r.y_true])
    p = np.array([lut[x] for x in r.y_pred])
    same_rows = np.arange(len(t))[None, :]
    assert abs(ev._macro_f1_resampled(t, p, same_rows)[0] - r.macro_f1) < 1e-12


def test_majority_baseline_is_neutral_and_scores_low_macro_f1():
    y_true = next(iter(fresh().values()))["true_label"].tolist()
    b = ev.majority_baseline(y_true)
    assert b.model_name == "always_neutral"
    assert b.macro_f1 < 0.15


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # report every failure, don't stop at the first
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
