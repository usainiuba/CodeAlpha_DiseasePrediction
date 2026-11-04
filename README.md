# CodeAlpha Disease Prediction

A machine learning project that estimates whether breast tissue cell measurements look **benign** or **malignant**, with a small Streamlit web app for trying it out. Built for the CodeAlpha Machine Learning Internship, **Task 4: Disease Prediction from Medical Data**.

> Educational project. It is not a medical tool and cannot diagnose anything. A real diagnosis needs a doctor, imaging and a lab test.

## Features

- Compares four classification models: Logistic Regression, SVM, Random Forest and XGBoost
- 5-fold cross-validation to choose the best model, then one final test on cases it has never seen
- Reports sensitivity and specificity, not only accuracy, because missing a malignant case is the costlier mistake
- Streamlit app with one-tap starting profiles, sliders for ten measurements, an animated result ring and probability bars
- "Where your values sit" view that compares each value with the typical benign and malignant case
- Mobile responsive design that respects the reduce-motion setting
- Unit tests and an end-to-end training check

## Dataset

Breast Cancer Wisconsin (Diagnostic), from the UCI Machine Learning Repository. It has 569 cases: 212 malignant and 357 benign. Each case is a set of measurements taken from a digital image of cells from a breast mass.

The data comes bundled with scikit-learn, so there is nothing to download and no account is needed.

Only the ten average measurements are used: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. In real life these come from software that analyses a lab image of the cells. In the app you set them with sliders, or load a typical profile, to see how the model reacts.

## Technology

Python, scikit-learn (models and metrics), XGBoost, pandas, NumPy, Streamlit (web app).

## Project structure

```
CodeAlpha_DiseasePrediction/
├── app.py            # Streamlit web app
├── train_model.py    # compares the models, tests the best, saves results
├── predict.py        # loads the model and scores one set of measurements
├── src/
│   ├── data.py       # loading, splitting, slider ranges
│   ├── models.py     # the four model pipelines
│   └── evaluate.py   # metrics and feature importance
├── ui/
│   ├── components.py # HTML pieces (header, ring, bars, tiles)
│   ├── labels.py     # readable names and form layout
│   └── style.css     # colours, layout, animations, mobile rules
├── tests/            # unit tests and an end-to-end check
├── models/           # trained model and results (created by training)
└── requirements.txt
```

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python train_model.py
```

This saves `models/disease_model.joblib`, `models/metrics.json`, `models/feature_options.json` and `models/classification_report.txt`. The saved model only works with the scikit-learn version it was trained with, so train again if you change versions.

## Run the web app

```powershell
python -m streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501).

## Run the tests

```powershell
python -m unittest discover -s tests -t .
```

## Model evaluation results

How the models were tested: 80% of the cases for training (455) and 20% held back for the final test (114), split so both parts keep the same share of malignant cases. The four models were compared with 5-fold cross-validation on the training part. The best one was then tested once on the held-back cases. Malignant is the positive class.

### Model comparison (5-fold cross-validation, ROC-AUC)

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.985 |
| **SVM** | **0.987** |
| Random Forest | 0.984 |
| XGBoost | 0.986 |

SVM was selected because it scored highest. All four models are almost tied, and the gaps are small enough to be noise on a dataset this size.

### Final test (114 unseen cases, threshold 0.5)

| Metric | Value |
|---|---|
| ROC-AUC | 0.985 |
| Accuracy | 95.6% |
| Sensitivity (malignant caught) | 95.2% |
| Specificity (benign cleared) | 95.8% |
| Precision | 93.0% |
| F1 | 0.941 |

| | Predicted benign | Predicted malignant |
|---|---|---|
| **Actually benign** | 69 | 3 |
| **Actually malignant** | 2 | 40 |

In plain words: of the 42 malignant cases the model caught 40 and missed 2. Of the 72 benign cases it wrongly flagged 3. In screening the missed cases matter most, and even two misses is too many for real use, which is why this stays a learning project.

## Known limitations

- Small, old dataset from one research setting, so results would not carry over to new hospitals or equipment.
- The model only sees ten numbers. It cannot look at a patient or an image.
- The final test has only 114 cases, so the numbers can move a little with a different split.
- Not a medical device. Never use it to make health decisions.

## Author

Usaini Uba, CodeAlpha Machine Learning Internship.
