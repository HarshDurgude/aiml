# Practical 4: Data Preprocessing, EDA and Train-Test Split
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Load Dataset (checks college PC path or local datasets/ folder)
path = "/home/avcoe/heart.csv" if os.path.exists("/home/avcoe/heart.csv") else "datasets/heart.csv"
df = pd.read_csv(path)

# 2. Data Overview & Preprocessing
print("Dataset Shape:", df.shape)
print("\nMissing Values Before Handling:\n", df.isnull().sum())

# Fill missing Age with median & drop duplicate records
if 'age' in df.columns:
    df['age'] = df['age'].fillna(df['age'].median())
df = df.drop_duplicates()
print("\nShape after removing duplicates:", df.shape)

# 3. Statistical Analysis
print("\n--- Summary Statistics ---")
print("Mean:\n", df.mean(numeric_only=True))
print("\nCorrelation:\n", df.corr(numeric_only=True))

# 4. Data Visualization (Histogram & Boxplot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=15, edgecolor='black')
plt.title("Age Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['chol'])
plt.title("Cholesterol Boxplot")
plt.tight_layout()
plt.show()

# 5. Train-Test Split (80:20)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Split Shapes ---")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
