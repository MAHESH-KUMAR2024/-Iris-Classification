from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree Model ---
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n🌳 Decision Tree")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Report:\n", classification_report(y_test, dt_pred, target_names=target_names))

# --- Logistic Regression Model ---
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n📈 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Report:\n", classification_report(y_test, lr_pred, target_names=target_names))
