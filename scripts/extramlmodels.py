import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE

# Load data
X = np.load("features/combined_features.npy")
y = np.load("features/labels.npy")

# Create directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# Scale and Balance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# Define all models and parameter grids
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    "ridge": (RidgeClassifier(), {
        "alpha": [0.1, 1.0, 10.0]
    }),
    "gb": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }),
    "et": (ExtraTreesClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }),
    "mlp": (MLPClassifier(max_iter=300, random_state=42), {
        "hidden_layer_sizes": [(100,), (64, 32)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.01]
    }),
    "ada": (AdaBoostClassifier(random_state=42), {
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1.0]
    }),
    "lgbm": (LGBMClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 64]
    }),
    "catboost": (CatBoostClassifier(verbose=0, random_state=42), {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6]
    })
}

# Loop through all models
for name, (model, param_grid) in models.items():
    print(f"\n🔍 Training {name.upper()}...")

    # Grid Search
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Save model
    model_path = f"outputs/models/{name}_model.pkl"
    joblib.dump(best_model, model_path)

    # Evaluate
    y_pred = best_model.predict(X_test)
    try:
        y_proba = best_model.predict_proba(X_test)[:, 1]
    except:
        y_proba = best_model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Control", "Dementia"])
    
    # Save report
    with open(f"outputs/reports/{name}_report.txt", "w") as f:
        f.write(f"{name.upper()} - Accuracy: {acc:.4f}\n")
        f.write(f"Best Parameters: {grid.best_params_}\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
    plt.title(f"{name.upper()} - Confusion Matrix")
    plt.savefig(f"outputs/results/{name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name.upper()} - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"outputs/results/{name}_roc_curve.png")
    plt.close()

    # Accuracy Bar
    plt.figure()
    plt.bar([name.upper()], [acc], color='skyblue')
    plt.ylim(0, 1)
    plt.title(f"{name.upper()} - Accuracy")
    plt.ylabel("Accuracy")
    plt.savefig(f"outputs/results/{name}_accuracy.png")
    plt.close()

print("\n✅ All models trained and results saved.")
