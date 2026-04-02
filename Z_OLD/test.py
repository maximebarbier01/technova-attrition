from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=808).fit(X, y)

clf.predict([X[8, :]])

clf.predict([X[13, :]])

y_hat_proba = clf.predict_proba(X)[:,1]
import seaborn as sns
sns.histplot(y_hat_proba)

y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)

accuracy = (193 + 346) / (193 + 346 + 19 + 11)

from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_pred)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, y_hat_proba)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)

precision_test, recall_test, thresholds_test = precision_recall_curve(
    y_test, y_scores_test
)

plt.plot(recall_test, precision_test, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Test Set")
plt.legend()
plt.show()

auc_test = auc(recall_test, precision_test)