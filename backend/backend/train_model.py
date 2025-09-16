import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

def generate_synthetic_data(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(50, 90, size=n)
    education_years = rng.randint(0, 20, size=n)
    mmse = rng.normal(27 - 0.05*(age-60), 2.0, size=n)
    recall_test = rng.normal(6 - 0.03*(age-60), 1.5, size=n)
    family_history = rng.binomial(1, 0.2, size=n)
    comorbid = rng.binomial(1, 0.25, size=n)
    risk_score = (0.03*(age-50) - 0.5*(mmse - 25) + 0.7*family_history +
                  0.4*comorbid - 0.02*education_years + rng.normal(0,0.5,size=n))
    y = (risk_score > np.percentile(risk_score, 70)).astype(int)

    return pd.DataFrame({
        "age": age,
        "education_years": education_years,
        "mmse": mmse,
        "recall_test": recall_test,
        "family_history": family_history,
        "comorbid": comorbid,
        "label": y
    })

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = generate_synthetic_data()
    X, y = df.drop(columns=["label"]), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

    joblib.dump({"model": model, "features": list(X.columns)}, "models/model.joblib")
    print("✅ Model saved to models/model.joblib")
