import numpy as np
import joblib
import os
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


X = np.load("features/combined_features.npy")
y = np.load("features/labels.npy")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Hyperparameter tuning
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Save
os.makedirs("outputs/models", exist_ok=True)
joblib.dump(model, "outputs/models/combined_feature_lasso.pkl")

# Predict
y_pred_cont = model.predict(X_test)
y_pred = np.clip(np.round(y_pred_cont), 0, 1)

# Evaluation
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Control", "Dementia"])
cm = confusion_matrix(y_test, y_pred)

# Save classification report
os.makedirs("outputs/reports", exist_ok=True)
with open("outputs/reports/lasso_classification_report.txt", "w") as f:
    f.write(f"Lasso - Accuracy: {acc:.4f}\n")
    f.write(f"Best Parameters: {grid.best_params_}\n\n")
    f.write(report)

# Save confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Control", "Dementia"], yticklabels=["Control", "Dementia"])
plt.title("Lasso - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("outputs/results", exist_ok=True)
plt.savefig("outputs/results/lasso_confusion_matrix.png")
plt.close()

# Save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_cont)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Lasso - ROC Curve")
plt.legend()
plt.savefig("outputs/results/lasso_roc_curve.png")
plt.close()

# Save accuracy bar
plt.figure()
plt.bar(["Lasso"], [acc], color='teal')
plt.ylim(0, 1)
plt.title("Lasso - Accuracy")
plt.savefig("outputs/results/lasso_accuracy.png")
plt.close()

# Print to console
print(f"Lasso Accuracy: {acc:.4f}")
print(f"Best Parameters: {grid.best_params_}")
