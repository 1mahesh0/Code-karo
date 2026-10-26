# Code-karo
Sentiment classifier for Amazon reviews (positive/neutral/negative) using TF-IDF + Logistic Regression, with an optional fine-tuned DistilBERT model. Includes NMF topic modeling to surface complaint themes from negative reviews, plus an interactive Streamlit demo app. Macro-F1: 0.68 on 100k reviews.
# Customer Review Sentiment Analysis (NLP)

Classify customer reviews as **positive / neutral / negative**, compare a classic TF-IDF model with a fine-tuned
**DistilBERT** transformer, discover the **main complaint themes** in negative reviews, and serve it in a **Streamlit** demo.

> **Business problem:** companies get thousands of reviews. This project scores them automatically and shows product
> teams *what* customers are unhappy about.

## Results  (fill in after you run it on the real data)

| Model | Macro-F1 | Negative F1 | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | _TBD_ | _TBD_ | `reports/baseline_metrics.json` |
| DistilBERT fine-tuned | _TBD_ | _TBD_ | `reports/transformer_metrics.json` |

Top complaint themes: _paste from `reports/complaint_themes.json`_

## Project structure
```
src/common.py             data loading, label mapping, text cleaning
src/train_baseline.py     TF-IDF + Logistic Regression, confusion matrix, top words
src/train_transformer.py  DistilBERT fine-tuning (GPU recommended)
src/topics.py             NMF topic modelling on negative reviews
app.py                    Streamlit demo
sample_data/              tiny synthetic data for smoke tests ONLY
run_smoke_test.sh         end-to-end check in seconds
```

## Dataset
Kaggle **Amazon Fine Food Reviews** (`Reviews.csv`, ~568k reviews): https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
Place it at `data/Reviews.csv`. Stars are mapped to labels: 1-2 = negative, 3 = neutral, 4-5 = positive.
Any CSV with `text,label` columns also works.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 0. (optional) check everything works on toy data
bash run_smoke_test.sh

# 1. Baseline (about 1-3 min on 100k rows)
python src/train_baseline.py --data data/Reviews.csv --max-rows 100000

# 2. Complaint themes
python src/topics.py --data data/Reviews.csv --max-rows 100000 --k 6

# 3. Transformer (use Google Colab free GPU; ~15 min for 30k rows)
python src/train_transformer.py --data data/Reviews.csv --max-rows 30000 --epochs 2

# 4. Demo app
streamlit run app.py
```
Note: use the same `--max-rows` for baseline and transformer to compare on a similar sample
(both use `random_state=42` and a stratified 80/20 split).

## Design decisions (good interview talking points)
- **Imbalance:** ~80% of reviews are positive, so I used `class_weight="balanced"` and report **macro-F1** and per-class F1, not accuracy.
- **Cleaning:** kept punctuation and negations ("not good") since they carry sentiment; removed HTML and URLs only.
- **Baseline first:** a TF-IDF model is fast and strong; the transformer must beat it to justify its cost.
- **Neutral class is hardest:** 3-star reviews are ambiguous; check the confusion matrix and error cases.
- **Topic modelling on negatives:** NMF over TF-IDF gives readable themes (delivery, packaging, taste, price).
- **Limitations:** star rating is a noisy label; sarcasm is hard; domain is food, so test before using on other products.

## Resume bullets (replace brackets with YOUR measured results)
- Built a 3-class sentiment classifier on [N] Amazon reviews; fine-tuned DistilBERT to reach [X] macro-F1 versus [Y] for a TF-IDF baseline.
- Applied NMF topic modelling to negative reviews, surfacing [N] recurring complaint themes (e.g. [theme]) for product teams.
- Deployed an interactive Streamlit app for real-time sentiment scoring with confidence scores.

## Interview questions to prepare
1. Why macro-F1 instead of accuracy? 2. TF-IDF vs embeddings? 3. Why does a transformer beat bag-of-words?
4. How did you handle the neutral class? 5. What would you do to deploy and monitor this?
