# Supervised Learning and Data Analysis Workshop

This repository contains my practical assignments for the **"Apprentissage et Analyse de Données" (Learning and Data Analysis)** course, completed during my **Master 1 in Computer Science at Université Lyon 1**.

The goal of these labs is to explore **supervised learning techniques** using Python and Scikit-learn — from preprocessing and feature selection to model comparison and pipeline automation.

---

## 🧩 Contents

### TP1 – Supervised Learning with Python
- Data preprocessing and feature engineering  
- Feature selection and normalization  
- Implementation of basic classifiers:
  - Decision Tree (CART)
  - k-Nearest Neighbors (KNN)
  - Multi-Layer Perceptron (MLP)
- Performance evaluation using:
  - Confusion matrix
  - Accuracy, Recall, Precision  
- Parameter tuning with **GridSearchCV**
- Building reusable **Scikit-learn pipelines**
- Model serialization with **Pickle**

### TP2 – Handling Heterogeneous Data
- Working with **mixed data types** (numerical + categorical)
- Managing **missing values** using different imputation strategies  
- Encoding categorical features with **OneHotEncoder**
- Normalizing continuous variables
- Comparative analysis of multiple classifiers:
  - Decision Tree (CART, ID3, Stump)
  - Random Forest
  - AdaBoost
  - Bagging
  - k-NN
  - MLP
  - XGBoost
- Model evaluation with **10-fold Cross Validation**, **Accuracy**, **Precision**, and **AUC**

---

## 🛠️ Tools & Libraries
- **Python 3.x**
- **NumPy**, **Pandas**, **Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **Jupyter Notebook**

---

## ⚙️ How to Run
Clone the repository and open the notebooks in Jupyter:

```bash
git clone https://github.com/yourusername/supervised-learning-lab.git
cd supervised-learning-lab
jupyter notebook
