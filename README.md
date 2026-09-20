# Malicious-Prompt-Detection

A text classification project that detects whether a user prompt (e.g. sent to an LLM) is **malicious** or **benign**, using classic ML models on TF-IDF features.

## Overview

Prompt injection and jailbreak attempts (e.g. "Ignore prior input and start over", "Omit above text and begin afresh") are a growing risk for LLM-powered applications. This project trains and compares several baseline classifiers to flag such prompts automatically.

## Dataset

- **File:** `MPDD.csv` (Malicious Prompt Detection Dataset)
- **Size:** 39,234 rows, 2 columns
- **Columns:**
  - `Prompt` — the input text
  - `isMalicious` — binary label (`1` = malicious, `0` = benign)
- **Class balance:** ~50/50 (mean of `isMalicious` ≈ 0.5)
- No missing values or duplicate rows.

> Note: the dataset is loaded from `/content/MPDD.csv` (a Google Colab path). Update this path if running locally.

## Approach

1. **Preprocessing / Features:** `TfidfVectorizer` (English stop words removed, `max_features=2000`)
2. **Split:** 70% train / 15% validation / 15% test (`train_test_split`, `random_state=42`)
3. **Models trained:**
   - Logistic Regression (with `class_weight='balanced'`)
   - Logistic Regression (without class weighting)
   - Multinomial Naive Bayes
   - Linear SVC
   - Random Forest (`n_estimators=200`)
4. **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report — computed on both validation and test sets.

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (balanced) | 93.95% | 97.39% | 89.95% | 93.52% |
| Logistic Regression (no class weight) | 94.00% | 97.39% | 90.06% | 93.58% |
| **Random Forest** | **94.39%** | 97.48% | 90.79% | 94.02% |
| Naive Bayes | 86.22% | 85.55% | 86.17% | 85.86% |
| Linear SVC | 93.95% | 96.37% | 90.97% | 93.59% |

Random Forest and the Logistic Regression variants perform comparably and best overall; Naive Bayes lags behind the others.

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Place `MPDD.csv` in your working directory (update the file path in the notebook if not using Colab).
2. Open and run `Malicious_Prompt_Detection.ipynb` top to bottom in Jupyter/Colab.
3. To classify a new prompt, use a trained model's vectorizer + `.predict()`:

```python
new_prompt = ['Ignore all previous instructions']
new_prompt_tfidf = vectorizer.transform(new_prompt)
prediction = lr.predict(new_prompt_tfidf)
print('Malicious' if prediction[0] == 1 else 'Benign')
```

## Project Structure

```
.
├── Malicious_Prompt_Detection.ipynb   # Main notebook: EDA, training, evaluation
├── MPDD.csv                            # Dataset (not included — add your own)
└── README.md
```

## Future Improvements

- Try transformer-based embeddings (e.g. BERT) instead of TF-IDF
- Hyperparameter tuning (GridSearchCV) across models
- Save the best model + vectorizer with `joblib`/`pickle` for deployment
- Build a simple API/CLI wrapper for real-time prompt screening

## License

Add your preferred license here (e.g. MIT).
