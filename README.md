Cell Analysis Screening (Disease Prediction)
A small project that learns to tell benign from malignant breast tissue cells from ten measurements, and shows the result in a web app.
I built it for Task 4 of the CodeAlpha Machine Learning Internship. The idea was to compare four classic models fairly, keep the best one, and be honest about what it can and cannot do.
This is a learning project. It is not a medical tool and cannot diagnose anything.
---
What it does
Loads the Breast Cancer Wisconsin (Diagnostic) dataset.
Compares four models: Logistic Regression, SVM, Random Forest and XGBoost.
Tests the best one once on cases it has never seen.
Lets you set ten measurements in a web app and see the model's estimate, with a short explanation of where your values sit compared with typical benign and malignant cases.
---
The data
569 cases, 212 malignant and 357 benign. Each case is a set of measurements taken from a digital image of cells from a breast mass.
The data comes bundled with scikit-learn, so there is nothing to download and no account to make.
I used only the ten average measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension). In real life these come from software that analyses a lab image of the cells. In the app you set them with sliders, or load a typical profile, to see how the model reacts.
---
How I tested it
80% of the cases for training (455) and 20% held back for the final test (114), split so both parts keep the same share of malignant cases.
The four models were compared with 5-fold cross-validation on the training part, using ROC-AUC.
The best model was then tested once on the held-back cases.
Malignant is the positive class. In screening, missing a malignant case is the costlier mistake, so I pay close attention to sensitivity (the share of malignant cases caught) as well as accuracy.
Comparison (5-fold cross-validation, ROC-AUC)
Model ROC-AUC
Logistic Regression 0.985
SVM 0.987
Random Forest 0.984
XGBoost 0.986
I picked SVM because it had the highest score. Honestly, all four models are almost tied, and the gaps are small enough to be noise on a dataset this size.
Final test (114 unseen cases, threshold 0.5)
Metric Value
ROC-AUC 0.985
Accuracy 95.6%
Sensitivity (malignant caught) 95.2%
Specificity (benign cleared) 95.8%
Precision 93.0%
F1 0.941
Confusion matrix:
Predicted benign Predicted malignant
Actually benign 69 3
Actually malignant 2 40
In plain words: of the 42 malignant cases the model caught 40 and missed 2. Of the 72 benign cases it wrongly flagged 3. In screening the missed cases matter most, and even 2 misses is 2 too many for real use, which is why this stays a learning project.
The full numbers are also saved in models/metrics.json, and the app shows them in the Model performance tab.
---
Run it yourself
You need Python 3.10 or newer. These commands are for Windows PowerShell.
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# 2. install the libraries
pip install -r requirements.txt
# 3. train (takes under a minute)
python train_model.py
# 4. open the web app
python -m streamlit run app.py
Training saves models/disease_model.joblib, metrics.json, classification_report.txt and feature_options.json.
The saved model only works with the scikit-learn version it was trained with. If you change versions, train again.
To run the tests:
python -m unittest discover -s tests -t .
---
The app
A header with the model's real test score.
One-tap starting profiles (typical benign, typical malignant, dataset average), then sliders for the ten measurements.
An animated result ring, a plain-language band (leans benign / uncertain / leans malignant) and two probability bars.
A "where your values sit" view that shows each key value between the typical benign and typical malignant value.
A performance tab with the model comparison, final test numbers and a confusion matrix.
Works on phones and respects the "reduce motion" setting.
---
Project structure
CodeAlpha_DiseasePrediction/
├── app.py            # the Streamlit web app
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
├── models/           # the trained model and results (created by training)
└── requirements.txt
---
Limitations
It is a small, old dataset from one research setting, so results would not carry over to new hospitals or equipment.
The model only sees ten numbers. It cannot look at a patient or an image.
The final test has only 114 cases, so the numbers can move a little with a different split.
A real diagnosis needs a doctor, imaging and a lab test.
---
Built with
Python, scikit-learn, XGBoost, pandas, NumPy and Streamlit.
---
Made by Usaini Uba as part of the CodeAlpha Machine Learning Internship.