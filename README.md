# Forge Workshop 01: Can AI Tell How You Feel?

**AI Ignite at York** | Friday 27 November 2026, 5:30 to 7:30 PM | WSC 101

We teach computers to read emotions in lines from *Friends* (the MELD dataset), three ways: train a small model (**BiLSTM**), fine-tune a big one (**BERT**), and simply ask an AI chatbot (**Gemini**). Then we score them all the same way and see who wins.

## Open the notebooks, in order

| Part | Who | What it does | Open |
|---|---|---|---|
| A | Sagar | Loads the dataset and the official split | [Open in Colab](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/forge_workshop_base.ipynb) |
| B1 | Ali | Trains BiLSTM and BERT (needs a T4 GPU) | [Open in Colab](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/notebooks/ali/Ali_1_Train_Models.ipynb) |
| B2 | Ali | Shows the BiLSTM and BERT results | [Open in Colab](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/notebooks/ali/Ali_2_See_Results.ipynb) |
| C | Davit | Asks Gemini (needs your free Gemini key) | [Open in Colab](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/notebooks/davit/davit_llm_prompting.ipynb) |
| D | Sheheryar | The scoreboard: every model, same test lines | [Open in Colab](https://colab.research.google.com/github/sagarh12/forge-workshop/blob/main/notebooks/sheheryar/evaluation.ipynb) |

In every notebook: **Runtime > Run all**. If Colab says "This notebook was not authored by Google", click **Run anyway**.

Every part has saved results, so if something breaks you can still move on to the next part.

## What's in this repo

```
forge_workshop_base.ipynb   Part A (Sagar). Stays here: the other notebooks rely on it
data/processed/             the cleaned MELD dataset (Mehwish)
notebooks/ali/              Part B1 and B2 (Ali)
notebooks/davit/            Part C (Davit)
notebooks/sheheryar/        Part D, the scoreboard (Sheheryar)
outputs/predictions/        saved results, one file per model (Ali and Davit)
outputs/figures/            charts and training logs (Ali and Davit)
tests/                      automatic checks for the scoreboard (Sheheryar)
```

Each folder has a short README that says who made which file.

## Rules for contributors

1. Use MELD's **official train / dev / test split**. The base notebook creates it for you.
2. Save results with `save_predictions()` (Step 7 of the base notebook), so the scoreboard can read them.
3. Everyone uses **their own free Gemini key** from [Google AI Studio](https://aistudio.google.com/apikey). Never put a key in a notebook or commit it.
4. Keep file names and folders as they are. The notebooks look for files in these exact places.
