import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load features and labels
X = np.load("features/combined_features.npy")
y = np.load("features/labels.npy")

# Create output directories
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# Apply Standard Scaler and SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

# Define base models
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, C=1.0, random_state=42)

# Voting Classifier (soft voting)
ensemble_model = VotingClassifier(estimators=[
    ('lasso', lasso),
    ('rf', rf),
    ('svm', svm)
], voting='soft') 

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save model
joblib.dump(ensemble_model, "outputs/models/ensemble_lasso_rf_svm.pkl")

# Predictions
y_pred = ensemble_model.predict(X_test)
y_proba = ensemble_model.predict_proba(X_test)[:, 1]

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Ensemble Accuracy: {acc:.4f}")

# Classification Report
report = classification_report(y_test, y_pred, target_names=["Control", "Dementia"])
with open("outputs/reports/ensemble_lasso_rf_svm_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.title("Ensemble - Confusion Matrix")
plt.savefig("outputs/results/ensemble_confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Ensemble - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("outputs/results/ensemble_roc_curve.png")
plt.close()

# Accuracy Plot
plt.figure()
plt.bar(["Ensemble (Lasso + RF + SVM)"], [acc], color='lightgreen')
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy - Ensemble Model")
plt.savefig("outputs/results/ensemble_accuracy.png")
plt.close()

print("Ensemble training complete and results saved.")
