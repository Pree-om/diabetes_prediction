# 🩺 Diabetes Prediction

## 📌 Overview

This project is a **Diabetes Prediction System** that uses machine learning techniques to predict whether a person is likely to have diabetes based on various health parameters. The model is trained on the **PIMA Indian Diabetes Dataset**, which contains medical information such as glucose levels, BMI, blood pressure, and more.

## ✨ Features

- 🔍 **Data Preprocessing**: Handling missing values, scaling features, and data cleaning.
- 📊 **Exploratory Data Analysis (EDA)**: Understanding data distribution and correlations.
- 🤖 **Machine Learning Models**: Implemented various classification models:
  - 📌 **K-Nearest Neighbors (KNN)**
  - 🌳 **Decision Tree Classifier**
  - 🌲 **Decision Tree with Max Depth = 9**
  - 🧠 **Multi-Layer Perceptron (MLP)**
  - 🏋️ **MLP using Standard Scaler**
  - 🎯 **Support Vector Machine (SVM)**
- 📈 **Model Evaluation**: Used accuracy, precision, recall, and F1-score to assess model performance.

## 📂 Dataset

The dataset used in this project is the **PIMA Indian Diabetes Dataset**, which contains the following features:

- 🤰 Pregnancies
- 🍬 Glucose
- ❤️ Blood Pressure
- 🩸 Skin Thickness
- 💉 Insulin
- ⚖️ BMI
- 🧬 Diabetes Pedigree Function
- 🎂 Age
- ✅ Outcome (1 = Diabetic, 0 = Non-Diabetic)

## 🛠 Installation

### 🔗 Prerequisites

Ensure you have the following installed:

- 🐍 Python 3.x
- 📓 Jupyter Notebook (optional)
- 📦 Required Python libraries (listed below)

### 📥 Install Dependencies

Run the following command to install necessary libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## 🚀 Usage

1. Clone this repository:

```bash
git clone https://github.com/Pree-om/diabetes_prediction.git
cd diabetes_prediction
```

2. Open and run the Jupyter Notebook:

```bash
jupyter notebook diabetes_prediction.ipynb
```

3. Train the model and evaluate performance.

## 📊 Model Performance

The following machine learning models were implemented and evaluated:

| 🤖 Model                         | 🎯 Training Accuracy | 📌 Testing Accuracy |
| --------------------------------- | ----------------- | ---------------- |
| 📌 K-Nearest Neighbors (KNN)      | 78.60%            | 72.44%           |
| 🌳 Decision Tree Classifier       | 100%              | 68.11%           |
| 🌲 Decision Tree (Max Depth = 9)  | 91.83%            | 69.29%           |
| 🧠 Multi-Layer Perceptron (MLP)   | 74.71%            | 66.14%           |
| 🏋️ MLP using Standard Scaler      | 83.07%            | 73.23%           |
| 🎯 Support Vector Machine (SVM)   | 83.66%            | 74.02%           |

## 🔮 Future Improvements

- 🔬 Improve feature selection for better accuracy.
- 🤖 Implement deep learning models (e.g., Neural Networks).
- 🌐 Deploy as a web application using Flask or Streamlit.
- 🏥 Incorporate additional health data for better predictions.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📜 License

This project is licensed under the **MIT License**.

## 🙌 Acknowledgments

- 📊 Dataset: [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 🎓 Inspired by various machine learning research studies on diabetes prediction.

---

**👨‍💻 Author:** [Pree-om](https://github.com/Pree-om)
