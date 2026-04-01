# -*- coding: utf-8 -*-

import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

file_path = "HTRU_2 2.csv"
df = pd.read_csv(file_path)

df.head(), df.shape

columns = [
    "mean_ip",
    "std_ip",
    "kurtosis_ip",
    "skewness_ip",
    "mean_dm_snr",
    "std_dm_snr",
    "kurtosis_dm_snr",
    "skewness_dm_snr",
    "class"
]

df = pd.read_csv(file_path, header=None, names=columns)


X = df.drop("class", axis=1)
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
df["class"].value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution (0 = Non-Pulsar, 1 = Pulsar)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

numeric_cols = df.columns[:-1]

plt.figure(figsize=(10, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    plt.hist(df[col], bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

corr = df.corr()
plt.figure()
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

feature = "skewness_ip"  # or any good one

plt.figure(figsize=(8, 5))

plt.hist(
    df[df["class"] == 0][feature],
    bins=40,
    alpha=0.6,
    label="Non-pulsar",
    density=True
)

plt.hist(
    df[df["class"] == 1][feature],
    bins=40,
    alpha=0.6,
    label="Pulsar",
    density=True
)

plt.xlabel(feature)
plt.ylabel("Density")
plt.title(f"Distribution of {feature} by Class")
plt.legend()
plt.show()

import seaborn as sns

feature = "skewness_ip"

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x=feature, hue="class", bins=40, kde=True)
plt.title(f"{feature} distribution by class")
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
axes = axes.flatten()

for i, feature in enumerate(df.columns[:-3]):
    sns.violinplot(data=df, x="class", y=feature, ax=axes[i])

    axes[i].set_title(feature.replace("_", " ").title(), fontsize=16)

    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Non-pulsar", "Pulsar"], fontsize=11)
    axes[i].tick_params(axis='y', labelsize=11)

fig.suptitle(
    "Feature Distributions",
    fontsize=20
)

plt.tight_layout()
plt.show()
