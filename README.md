
# 🧠 Stroke Prediction MLOps Project

A full-cycle machine learning pipeline to predict stroke occurrences using a classification model.  
This project applies MLOps principles including training, tracking, deployment, and monitoring.

---

## 🚀 Features

- ⚕️ Predict whether a person is at risk of stroke
- 📊 Trained on real-world stroke dataset
- 🧪 Model evaluation and logging with MLflow
- 🔌 FastAPI for REST API access
- 📈 Monitoring dashboard with Streamlit
- 🧠 SMOTE applied to handle class imbalance

---

## 🧰 Tech Stack

- Python 3.10+
- scikit-learn
- Pandas, NumPy
- imbalanced-learn (SMOTE)
- FastAPI
- Streamlit
- MLflow
- Joblib

---

## 📁 Project Structure

```bash
stroke-prediction-mlops/
├── app.py               # FastAPI REST API
├── streamlit_app.py     # Monitoring dashboard
├── train.py             # Model training and MLflow logging
├── model/               # Folder for saved model.pkl
├── logs/                # Folder for prediction logs
├── stroke-data.csv      # Dataset used for training
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## 🎯 Input Features

| Feature            | Type   | Description                                   |
|--------------------|--------|-----------------------------------------------|
| `gender`           | int    | Encoded gender (0 = Female, 1 = Male)         |
| `age`              | float  | Age of the person                             |
| `hypertension`     | int    | 1 if patient has hypertension                 |
| `heart_disease`    | int    | 1 if patient has heart disease                |
| `ever_married`     | int    | 1 if patient was ever married                 |
| `work_type`        | int    | Encoded work type                             |
| `Residence_type`   | int    | 0 = Rural, 1 = Urban                          |
| `avg_glucose_level`| float  | Average glucose level                         |
| `bmi`              | float  | Body Mass Index                               |
| `smoking_status`   | int    | Encoded smoking status                        |

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gurusingagerry03/stroke-prediction-mlops.git
cd stroke-prediction-mlops
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python train.py
```

This script will:
- Clean the dataset
- Apply label encoding
- Apply SMOTE
- Train a RandomForestClassifier
- Log results to MLflow
- Save the model to `model/model.pkl`

---

## 🔌 Run the FastAPI Prediction API

```bash
uvicorn app:app --reload
```

Visit Swagger UI to test:
http://127.0.0.1:8000/docs

---

## 📊 Launch the Monitoring Dashboard

```bash
streamlit run streamlit_app.py
```

You will see:
- Total predictions made
- Distribution of stroke vs non-stroke
- Daily prediction trends
- Average age and glucose level per group

---

## 🧪 Example JSON Input

```json
{
  "gender": 1,
  "age": 67,
  "hypertension": 1,
  "heart_disease": 0,
  "ever_married": 1,
  "work_type": 2,
  "Residence_type": 1,
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": 1
}
```

---

## 👤 Author

Built with ❤️ by **Gerry0303**

GitHub: [@gurusingagerry03](https://github.com/gurusingagerry03)
