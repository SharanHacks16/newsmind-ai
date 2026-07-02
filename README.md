# NewsMind AI

An intelligent news category classification system using a locally trained machine learning model.

## Features

- React, Vite, Tailwind CSS frontend
- Flask API backend with CORS enabled
- Local scikit-learn ML pipeline
- Text preprocessing with lowercasing, punctuation removal, stop-word removal, and stemming
- TF-IDF vectorization
- Logistic Regression, Multinomial Naive Bayes, and Linear SVM training
- Automatic best-model selection by test accuracy
- Evaluation graphs for reports and viva presentation
- No online AI API or API key

## Run Backend

```bash
cd NewsMindAI/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
python evaluate.py
python app.py
```

## Run Frontend

```bash
cd NewsMindAI/frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://127.0.0.1:5173`.

## Dataset

Use the BBC News Dataset for final training. Put the CSV in `dataset/` as `bbc_news.csv`, `bbc-news-data.csv`, or `BBC News Train.csv`. The included sample CSV lets the app run immediately.

## API

POST `http://127.0.0.1:5000/api/predict`

```json
{"text":"Paste a news article here"}
```

Response:

```json
{"category":"Sports","confidence":98.72,"prediction_time_ms":12.4}
```
