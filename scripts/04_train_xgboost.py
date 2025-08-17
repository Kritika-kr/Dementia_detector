import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load features and labels
X = np.load("features/combined_features.npy")
y = np.load("features/labels.npy")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid = GridSearchCV(xgb, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Save model
os.makedirs("outputs/models", exist_ok=True)
joblib.dump(best_model, "outputs/models/combined_feature_xgboost_tuned.pkl")

# Predictions
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Accuracy & Classification Report
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Control", "Dementia"])

# Save report
os.makedirs("outputs/reports", exist_ok=True)
with open("outputs/reports/xgboost_classification_report.txt", "w") as f:
    f.write(f"XGBoost (Tuned) - Accuracy: {acc:.4f}\n")
    f.write(report)

# Save accuracy
with open("outputs/reports/xgboost_accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Control", "Dementia"],
            yticklabels=["Control", "Dementia"])
plt.title("XGBoost (Tuned) - Confusion Matrix")
os.makedirs("outputs/results", exist_ok=True)
plt.savefig("outputs/results/xgboost_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost (Tuned) - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("outputs/results/xgboost_roc_curve.png")
plt.close()

# Accuracy Bar Plot
plt.figure()
plt.bar(["XGBoost (Tuned)"], [acc], color='orange')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.savefig("outputs/results/xgboost_accuracy.png")
plt.close()
