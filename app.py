import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    ################ Step 1 Create Web Title #####################
    st.title("Binary Classification Streamlit App")
    st.sidebar.title("Binary Classification Streamlit App")
    st.markdown("เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    st.sidebar.markdown("เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")

    ############### Step 2 Load dataset and Preprocessing data ##########
    @st.cache_data(persist=True)
    def load_data():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            st.error("File not found. Please check the file path.")
            return None

        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list, model, x_test, y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=['edible', 'poisonous'])
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    df = load_data()
    if df is None:
        return

    x_train, x_test, y_train, y_test = spliting_data(df)

    st.sidebar.subheader("Choose Classifiers")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    ############### Step 3 Train a SVM Classifier ##########
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics, model, x_test, y_test)

    ############### Step 4 Training a Logistic Regression Classifier ##########
    elif classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression results")
            model = LogisticRegression(C=C, max_iter=1000)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics, model, x_test, y_test)

    ############### Step 5 Training a Random Forest Classifier ##########
    elif classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, 10)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, None)  # None for unlimited depth

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics, model, x_test, y_test)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset")
        st.write(df)
#hhh
if __name__ == '__main__':
    main()