import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X = np.load("features/combined_features.npy")
y = np.load("features/labels.npy")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# SVM + Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(probability=True, random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Save model
os.makedirs("outputs/models", exist_ok=True)
joblib.dump(best_model, "outputs/models/combined_feature_svm_tuned.pkl")

# Predict
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Control", "Dementia"])

# Save report
os.makedirs("outputs/reports", exist_ok=True)
with open("outputs/reports/svm_classification_report.txt", "w") as f:
    f.write(f"SVM (RBF) (Tuned) - Accuracy: {acc:.4f}\n")
    f.write(report)
with open("outputs/reports/svm_accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Control", "Dementia"],
            yticklabels=["Control", "Dementia"])
plt.title("SVM (RBF) (Tuned) - Confusion Matrix")
os.makedirs("outputs/results", exist_ok=True)
plt.savefig("outputs/results/svm_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM (RBF) (Tuned) - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("outputs/results/svm_roc_curve.png")
plt.close()

# Accuracy bar
plt.figure()
plt.bar(["SVM (Tuned)"], [acc], color='lightgreen')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.savefig("outputs/results/svm_accuracy.png")
plt.close()
