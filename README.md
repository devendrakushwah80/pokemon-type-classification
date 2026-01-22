# Pokémon Type Classification using Machine Learning

This project focuses on predicting the **Pokémon Type** (such as Fire, Water, Grass, etc.) using machine learning techniques.
Multiple classification models are trained and compared to identify the best-performing algorithm.

The project demonstrates a complete **end-to-end machine learning workflow** including preprocessing, model training, hyperparameter tuning, and evaluation.

---

## 📂 Dataset

- **File:** `pokemon_stats_2025.csv`
- **Description:**  
  The dataset contains Pokémon attributes such as combat-related features and numerical characteristics.
- **Target Variable:** `type` (Pokémon Type – categorical)

---

## 🧠 Machine Learning Models Used

The following classifiers were implemented using **Scikit-learn Pipelines**:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gradient Boosting Classifier  

Each model was tuned using **GridSearchCV** to obtain optimal hyperparameters.

---

## ⚙️ Data Preprocessing

A robust preprocessing pipeline was built using `ColumnTransformer`:

- **Numerical Features**
  - StandardScaler
- **Categorical Features**
  - OneHotEncoder
- **Pipeline**
  - Preprocessing + Model combined to avoid data leakage

---

## 📊 Model Evaluation Metrics

Models were evaluated on the test dataset using:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- ROC-AUC (where applicable)

A final comparison table helps determine the best classifier for Pokémon type prediction.

---

## 📁 Project Structure

├── pokemon_type_classification.ipynb
├── pokemon_stats_2025.csv
├── requirements.txt
└── README.md


---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/pokemon-type-classification.git
cd pokemon-type-classification
```
2. Install dependencies

pip install -r requirements.txt


3. Launch the notebook

jupyter notebook pokemon_type_classification.ipynb

🎯 Key Highlights

Multi-class classification problem

Clean preprocessing using Pipelines

Hyperparameter tuning with GridSearchCV

Comparison of multiple ML algorithms

Interview-ready project structure

🧑‍💻 Author

Devendra Kushwah
Aspiring Machine Learning Engineer

⭐ If you find this project useful, feel free to star the repository!
