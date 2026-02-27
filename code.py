import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame

df = df.replace("?", pd.NA)
df = df.dropna()

X = df.drop("class", axis=1)
y = df["class"]

X = pd.get_dummies(X)
y = y.map({">50K": 1, "<=50K": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        verbosity=0
    )
}

accuracy_list = []
precision_list = []
recall_list = []

for name, model in models.items():

    if name in ["SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))

x = range(len(models))

plt.figure()

plt.bar([i - 0.25 for i in x], accuracy_list, width=0.25,
        color="magenta", edgecolor="black", linewidth=2, label="Accuracy")

plt.bar(x, precision_list, width=0.25,
        color="lime", edgecolor="black", linewidth=2, label="Precision")

plt.bar([i + 0.25 for i in x], recall_list, width=0.25,
        color="gray", edgecolor="black", linewidth=2, label="Recall")

plt.xticks(x, models.keys())
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Adult Income Dataset - Model Comparison")
plt.legend()

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.tick_params(width=2)

plt.show()

for i, model in enumerate(models.keys()):
    print(f"{model}:")
    print(" Accuracy :", accuracy_list[i])
    print(" Precision:", precision_list[i])
    print(" Recall   :", recall_list[i])
    print()
