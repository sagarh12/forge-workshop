# Predictions: one file per model

The scoreboard (Part D) reads every file in this folder. Please keep these names and this location.

| File | Made by | Test lines |
|---|---|---|
| `bilstm_predictions.csv` | Ali (Part B) | all 2,610 |
| `bert_finetuned_predictions.csv` | Ali (Part B) | all 2,610 |
| `zero_shot_predictions.csv` | Davit (Part C) | first 100 |
| `few_shot_predictions.csv` | Davit (Part C) | first 100 |
| `balanced_few_shot_predictions.csv` | Davit (Part C) | first 100 |

Every file has the same columns: `utterance`, `true_label`, `predicted_label`, `model_name`.
