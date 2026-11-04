"""Helpers shared by the notebook and the Streamlit app."""
import numpy as np
import pandas as pd

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]


def zeros_to_nan(X):
    """In this dataset a 0 in Glucose/BloodPressure/SkinThickness/Insulin/BMI is
    physiologically impossible, so it means 'not measured'. Turn it into NaN."""
    X = pd.DataFrame(X, columns=FEATURES).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X
