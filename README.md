# 🧠 Supervised Learning and Data Analysis Workshop

This repository contains my practical assignments for the **"Apprentissage et Analyse de Données" (Learning and Data Analysis)** course, completed during my **Master 1 in Computer Science at Université Claude Bernard Lyon 1**.

The goal of these workshops is to explore various **supervised learning** techniques using **Python** and **Scikit-learn**, from data preprocessing and feature engineering to model evaluation and optimization.

---

## 📘 Overview

These notebooks demonstrate the end-to-end workflow of a typical machine learning project:

1. Data preprocessing and cleaning  
2. Feature engineering and selection  
3. Model training and evaluation  
4. Hyperparameter optimization  
5. Pipeline creation and model persistence  
6. Handling heterogeneous and incomplete datasets  

Each lab builds on the previous one to gradually introduce advanced data analysis techniques.

---

## 🧩 Contents

### 🧮 TP1 – Supervised Learning with Python
Focus: feature engineering, normalization, and model comparison.

**Main steps:**
- Load and analyze credit scoring data  
- Split into training and testing sets  
- Train and compare three classifiers:
  - Decision Tree (CART)
  - k-Nearest Neighbors (k=5)
  - Multi-Layer Perceptron (40–20 hidden layers)
- Evaluate performance using:
  - Confusion matrix  
  - Accuracy, Recall, and Precision  
- Normalize continuous features (StandardScaler / MinMaxScaler)
- Perform feature importance ranking with **RandomForestClassifier**
- Tune parameters using **GridSearchCV**
- Build a **Scikit-learn pipeline** combining preprocessing and classification
- Save and reload the trained model using **Pickle**

---

### 🧭 TP2 – Learning with Heterogeneous Data
Focus: handling categorical data, missing values, and model generalization.

**Main steps:**
- Use the **Credit Approval Dataset** from UCI ML Repository  
- Handle missing values:
  - Replace '?' with `np.nan`
  - Impute numeric columns with the mean  
  - Impute categorical columns with the most frequent value  
- Encode categorical variables using **OneHotEncoder**
- Normalize numerical features with **StandardScaler**
- Compare multiple classifiers with 10-fold cross-validation:
  - Decision Tree (CART, ID3, Stump)
  - k-Nearest Neighbors  
  - Multi-Layer Perceptron  
  - Random Forest  
  - Bagging  
  - AdaBoost  
  - XGBoost  
- Evaluate using:
  - **Accuracy**
  - **Precision**
  - **AUC (Area Under ROC Curve)**
  - **Execution time**
- Analyze the influence of normalization and hyperparameters

---

## ⚙️ How to Run

Clone this repository and open the notebooks in Jupyter:

```bash
git clone https://github.com/yourusername/supervised-learning-lab.git
cd supervised-learning-lab
jupyter notebook
```

### 🧰 Requirements
Make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
pandas
matplotlib
scikit-learn
xgboost
jupyter
```

---

## 📊 Key Learning Outcomes

- Understand the workflow of **supervised learning**  
- Apply data preprocessing techniques (scaling, encoding, imputation)  
- Compare the behavior of different classifiers  
- Use **cross-validation** and **metrics** for model evaluation  
- Perform **feature selection** and **hyperparameter tuning**  
- Automate training using **Scikit-learn Pipelines**  
- Work with **heterogeneous datasets** and missing data  

---

## 🧠 Course Information

- **Course name:** Apprentissage et Analyse de Données (Learning and Data Analysis)  
- **Level:** Master 1 – Computer Science  
- **University:** Université Claude Bernard Lyon 1  
- **Instructors:** Haytham Elghazel, Ichrak Ennaceur, Hana Sebia  
- **Academic year:** 2024–2025  

---

## 🪄 Technologies Used

| Category | Tools |
|-----------|--------|
| Programming Language | Python 3.x |
| Libraries | NumPy, Pandas, Scikit-learn, XGBoost |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |
| Model Export | Pickle |

---

## 📈 Example Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

# Example pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(40, 20), random_state=1))
])

model.fit(X_train, y_train)

# Save pipeline
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load and reuse
with open('credit_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    predictions = loaded_model.predict(X_test)
```

---

## 👨‍💻 Author

**Alexandre Mantador**  
Master’s student in Computer Science – *Image, Développement et Technologie 3D (ID3D)*  
Université Claude Bernard Lyon 1  

📍 Lyon, France  
🔗 [GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📜 License

This repository is intended for **educational and research purposes** only.  
You are free to reuse and adapt the notebooks with proper attribution.

---

> *“Learning from data is the first step toward making sense of the world.”*
