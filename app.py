import streamlit as st
import pandas as pd
import os
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, predict_model as cls_predict_model, save_model as cls_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, predict_model as reg_predict_model, save_model as reg_save_model
import pickle
from ydata_profiling import ProfileReport
def drop_problematic_columns(df, target_col):
    df = df.copy()
    threshold = 0.5 * len(df)
    df = df.loc[:, df.isnull().sum() < threshold]

    for col in df.columns:
        if df[col].nunique() <= 1 and col != target_col:
            df.drop(columns=[col], inplace=True)

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().sample(min(10, df[col].dropna().shape[0]), random_state=1) if df[col].dropna().shape[0] > 0 else []
            if any(not isinstance(x, (str, int, float, bool, type(None))) for x in sample_vals):
                df.drop(columns=[col], inplace=True)

    return df

with st.sidebar:
    st.title("AutoML")
    choice = st.radio("Navigation", ['Upload', 'Clean Data', 'Profiling', 'ML', 'Download'])
    st.info("This application allows you to build an automated ML pipeline")

raw_file = "raw_data.csv"
cleaned_file = "sourcedata.csv"

df = pd.read_csv(cleaned_file) if os.path.exists(cleaned_file) else None

if choice == "Upload":
    st.title("Upload Your Raw Dataset!")
    file = st.file_uploader("Upload Your dataset", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        df.to_csv(raw_file, index=False)
        df.to_csv(cleaned_file, index=False)
        st.success("File uploaded successfully!")
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df)

if choice == "Clean Data":
    st.title("Auto Clean Your Dataset")

    if df is not None:
        st.subheader("Raw Data")
        st.dataframe(df)

        if st.button("Run Cleaning"):
            cleaned_df = df.copy()

            
            thresh = len(cleaned_df) * 0.5
            cleaned_df = cleaned_df.loc[:, cleaned_df.isnull().sum() < thresh]

            
            num_cols = cleaned_df.select_dtypes(include=['number']).columns
            for col in num_cols:
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)

            
            cat_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col].fillna(mode_val[0], inplace=True)

            st.success("Data cleaned successfully!")
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_df)

            cleaned_df.to_csv(cleaned_file, index=False)

    else:
        st.warning("Please upload a dataset first.")

import sweetviz as sv

if choice == "Profiling":
    st.title("Data Profiling Report")

    if df is not None:
        st.info("Generating profiling report. Please wait...")

        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
        report_path = "profiling_report.html"
        profile.to_file(report_path)

        with open(report_path, 'r', encoding='utf-8') as f:
            html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

    else:
        st.warning("Please upload or clean data first.")


if choice == "ML":
    st.title("AutoML Module")
    st.info("This section will let you build and compare models automatically.")

    if df is not None:
        task_type = st.radio("Choose ML Task", ["Classification", "Regression"])
        target = st.selectbox("Select your target column", df.columns)

        if st.button("Train and Compare Models"):
            if task_type == "Classification":
                cls_setup(df, target=target, verbose=False)
                setup_df = cls_pull()
                st.info("Classification setup summary")
                st.dataframe(setup_df)

                best_model = cls_compare_models()
                compare_df = cls_pull()
                st.info("Best classification model")
                st.dataframe(compare_df)

                predictions = cls_predict_model(best_model)
                predictions.to_csv("predictions.csv", index=False)
                compare_df.to_csv("compare_report.csv", index=False)
                cls_save_model(best_model, "best_model")

                st.success("Classification model trained and saved!")

            elif task_type == "Regression":
                reg_setup(df, target=target, verbose=False)
                setup_df = reg_pull()
                st.info("Regression setup summary")
                st.dataframe(setup_df)

                best_model = reg_compare_models()
                compare_df = reg_pull()
                st.info("Best regression model")
                st.dataframe(compare_df)

                predictions = reg_predict_model(best_model)
                predictions.to_csv("predictions.csv", index=False)
                compare_df.to_csv("compare_report.csv", index=False)
                reg_save_model(best_model, "best_model")

                st.success("Regression model trained and saved!")
    else:
        st.warning("Please upload or clean data first.")
        
if choice == "Download":
    st.title("Download Options")
    st.info("This section will let you download ML model, reports and predictions.")
    model_path = "best_model.pkl"

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        st.download_button(
            label="Download Best Model (.pkl)",
            data=model_bytes,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )
    else:
        st.warning("No trained model found. Please train a model first in the ML section.")
    if os.path.exists("predictions.csv"):
        with open("predictions.csv", "rb") as f:
            st.download_button(
                label="Download Predictions (.csv)",
                data=f,
                file_name="predictions.csv",
                mime="text/csv"
            )
    else:
        st.warning("No predictions found. Please train a model first.")
    if os.path.exists("compare_report.csv"):
        with open("compare_report.csv", "rb") as f:
            st.download_button(
                label="Download Compare Report (.csv)",
                data=f,
                file_name="compare_report.csv",
                mime="text/csv"
            )
    else:
        st.warning("No report found. Please train a model first.")

