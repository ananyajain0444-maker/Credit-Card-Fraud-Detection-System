import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from src.utils import (
    create_folders,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve
)

def run_training():
    print("🚀 Starting Training...")

    # Create folders
    create_folders()

    # Load data
    df = pd.read_csv("data/creditcard.csv")
    print("✅ Data Loaded:", df.shape)

    # Plot class distribution
    plot_class_distribution(df)

    # Split features/target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Handle imbalance
    print("⚖️ Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Train model
    print("🤖 Training Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Generate 3 more images
    plot_confusion_matrix(y_test, y_pred)
    plot_precision_recall(y_test, y_proba)
    plot_roc_curve(y_test, y_proba)

    # Save model
    joblib.dump(model, "models/fraud_model.pkl")

    print("✅ Model Saved")
    print("🖼️ 4 Images Generated in outputs/")

if __name__ == "__main__":
    run_training()