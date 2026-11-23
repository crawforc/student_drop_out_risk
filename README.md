# Student Dropout Risk Prediction Tool

WGU BS Data Analytics Capstone - Cindy Crawford

## Overview
This project uses machine learning to predict whether a student is at risk of dropping out, so schools can offer support early. It includes data exploration, a trained Random Forest model, a fairness report, and a FastAPI app that returns a prediction for a student.

**Research question:** Can student academic, financial, and wellbeing data be used to accurately predict whether a student will drop out?

## Data
"Student Academic Performance and Dropout Risk" by Mohan Krishna Thalla, Kaggle, CC0 Public Domain license:
https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk

## Project Structure
| File / Folder | Purpose |
|---|---|
| `data/` | Original dataset, cleaned dataset, processed preview |
| `eda.py` | Data checks, cleaning, summary statistics, charts |
| `preview_processed.py` | Saves a preview of the data after one-hot encoding |
| `ml/data.py` | Selects features and prepares data for the model |
| `ml/model.py` | Train, score, predict, save/load, and fairness functions |
| `train_model.py` | Trains and saves the model, creates performance and fairness reports |
| `model/` | Saved model and encoder |
| `main.py` | FastAPI app |
| `local_api.py` | Sends test requests to the running API |
| `charts/` | Exploratory and performance charts |
| `screenshots/` | Evidence of each step |
| `fairness_report.csv` | Model performance for each student group |
| `model_card.md` | Model use, performance, ethics, and limits |

## How to Run (Windows)
1. Create and turn on a virtual environment, then install packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Explore and clean the data:
   ```
   python eda.py
   ```
3. Train the model and create the reports:
   ```
   python train_model.py
   ```
4. Start the API, then open http://127.0.0.1:8000/docs to test it:
   ```
   uvicorn main:app --reload
   ```
5. In a second terminal, send a test request:
   ```
   python local_api.py
   ```

## Results
| Metric | Score |
|---|---|
| Accuracy | 80.8% |
| Precision | 66.5% |
| Recall | 80.9% |
| F1 | 73.0% |

Semester 1 and semester 2 GPA were the strongest predictors. See `model_card.md` for fairness results, ethical considerations, and limitations.
