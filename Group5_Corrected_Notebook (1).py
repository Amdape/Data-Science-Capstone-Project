# ============================================================
# TS Academy Data Science Capstone — Group 5
# Title: Analysis of the Effect of Screen Time on Stress Level
# Track: Supervised Learning — Binary Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. LOAD DATASET ──────────────────────────────────────────
df_smartphone = pd.read_csv('/content/Smartphone_Usage_Productivity_Dataset_50000 (1).csv')
print("Raw shape:", df_smartphone.shape)
df_smartphone.head()

# ── 2. DATA CLEANING ─────────────────────────────────────────

# Step 1: Rename column
df_smartphone = df_smartphone.rename(columns={'Daily_Phone_Hours': 'Average_Daily_Phone_Hours'})

# Step 2: Check data info
df_smartphone.info()

# Step 3: Check for null values
print("Null values:\n", df_smartphone.isnull().sum())

# Step 4: Check for duplicates
print("Duplicates:", df_smartphone.duplicated().sum())

# Step 5: Flag and remove logical inconsistencies
# Social_Media_Hours cannot exceed Average_Daily_Phone_Hours
df_smartphone["Invalid_Daily_Phone_Hours"] = (
    df_smartphone["Social_Media_Hours"] > df_smartphone["Average_Daily_Phone_Hours"]
)
pct_invalid = df_smartphone["Invalid_Daily_Phone_Hours"].mean() * 100
print(f"Logically inconsistent rows: {pct_invalid:.1f}%")

df_smartphone = df_smartphone[
    df_smartphone['Social_Media_Hours'] < df_smartphone['Average_Daily_Phone_Hours']
]
df_smartphone.drop('Invalid_Daily_Phone_Hours', axis=1, inplace=True)
print("Clean shape:", df_smartphone.shape)

# Step 6: Set index
df_smartphone = df_smartphone.set_index('User_ID')

# Step 7: Feature engineering — composite weekly screen time
df_smartphone['Total_Weekly_Screen_Time'] = (
    df_smartphone['Social_Media_Hours'] * 5
) + df_smartphone['Weekend_Screen_Time_Hours']

df_smartphone.head()
df_smartphone.describe()

# ── 3. EXPLORATORY DATA ANALYSIS ─────────────────────────────

# Numerical and categorical splits
df_num = df_smartphone.select_dtypes(include="number")
df_cat = df_smartphone.select_dtypes(exclude="number")

# Stress Level distribution
print("\nStress Level value counts:")
print(df_smartphone['Stress_Level'].value_counts().sort_index())

plt.figure(figsize=(8, 4))
sns.countplot(data=df_smartphone, x='Stress_Level', palette='coolwarm')
plt.title('Distribution of Stress Level (10 Classes)')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df_smartphone.corr(numeric_only=True)
plt.figure(figsize=(12, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix — Numerical Features')
plt.tight_layout()
plt.show()

# Distribution plots — all numerical features
numerical_cols = ['Age', 'Average_Daily_Phone_Hours', 'Social_Media_Hours',
                  'Work_Productivity_Score', 'Sleep_Hours',
                  'Caffeine_Intake_Cups', 'Weekend_Screen_Time_Hours', 'App_Usage_Count']

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_smartphone, x=col, kde=True, color='steelblue')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# Age group segmentation
df_smartphone['Age_Group'] = pd.cut(
    df_smartphone['Age'],
    bins=[0, 20, 30, 40, 70],
    labels=['Under 20', '21-30', '31-40', '41-70']
)

# Gender and Occupation groupings
df_male      = df_smartphone[df_smartphone["Gender"] == 'Male']
df_female    = df_smartphone[df_smartphone["Gender"] == 'Female']
df_other     = df_smartphone[df_smartphone["Gender"] == 'Other']

df_Business_Owner = df_smartphone[df_smartphone["Occupation"] == 'Business Owner']
df_Freelancer     = df_smartphone[df_smartphone["Occupation"] == 'Freelancer']
df_Professional   = df_smartphone[df_smartphone["Occupation"] == 'Professional']
df_Student        = df_smartphone[df_smartphone["Occupation"] == 'Student']

# ── GROUP STRESS ANALYSIS ─────────────────────────────────────
print("\n=== MEAN STRESS BY GENDER ===")
print(df_smartphone.groupby('Gender')['Stress_Level'].mean().sort_values(ascending=False).round(2))

print("\n=== MEAN STRESS BY OCCUPATION ===")
print(df_smartphone.groupby('Occupation')['Stress_Level'].mean().sort_values(ascending=False).round(2))

print("\n=== MEAN STRESS BY AGE GROUP ===")
print(df_smartphone.groupby('Age_Group', observed=True)['Stress_Level'].mean().sort_values(ascending=False).round(2))

# Visualise group stress
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df_smartphone.groupby('Gender')['Stress_Level'].mean().sort_values().plot(
    kind='bar', ax=axes[0], color='steelblue', title='Mean Stress by Gender')
df_smartphone.groupby('Occupation')['Stress_Level'].mean().sort_values().plot(
    kind='bar', ax=axes[1], color='coral', title='Mean Stress by Occupation')
df_smartphone.groupby('Age_Group', observed=True)['Stress_Level'].mean().sort_values().plot(
    kind='bar', ax=axes[2], color='green', title='Mean Stress by Age Group')

for ax in axes:
    ax.set_ylabel('Mean Stress Level')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# ── 4. FEATURE ENGINEERING & ENCODING ────────────────────────

# !! CRITICAL FIX: Binarise the target variable !!
# Stress Level 1-5 = Low Stress (0) | Stress Level 6-10 = High Stress (1)
df_smartphone['Stress_Binary'] = (df_smartphone['Stress_Level'] > 5).astype(int)

print("\nBinary target distribution:")
print(df_smartphone['Stress_Binary'].value_counts())
print(df_smartphone['Stress_Binary'].value_counts(normalize=True).round(3))

plt.figure(figsize=(6, 4))
sns.countplot(data=df_smartphone, x='Stress_Binary', palette='Set2')
plt.title('Binary Stress Distribution')
plt.xticks(ticks=[0, 1], labels=['Low Stress (1-5)', 'High Stress (6-10)'])
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# One-Hot Encoding
df_encoded = pd.get_dummies(
    df_smartphone,
    columns=['Gender', 'Occupation', 'Device_Type'],
    drop_first=False,
    dtype=int
)

# Drop helper columns not needed for modelling
df_encoded = df_encoded.drop(columns=['Age_Group', 'Stress_Level'], errors='ignore')

# Feature matrix and target
X = df_encoded.drop(columns=['Stress_Binary'])
y = df_encoded['Stress_Binary']

print(f"\nFeature matrix: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ── 5. TRAIN-TEST SPLIT ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 6. BASELINE MODEL (rf1) ───────────────────────────────────
rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train, y_train)

y_pred1 = rf1.predict(X_test)

print("\n" + "="*55)
print("BASELINE MODEL (rf1) RESULTS")
print("="*55)
print(f"Training Accuracy: {rf1.score(X_train, y_train):.4f}")
print(f"Test Accuracy:     {accuracy_score(y_test, y_pred1):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred1, target_names=['Low Stress', 'High Stress']))

# Confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(6, 4))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Stress', 'High Stress'],
            yticklabels=['Low Stress', 'High Stress'])
plt.title('Confusion Matrix — Baseline Model (rf1)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ── 7. TUNED MODEL (rf2) ──────────────────────────────────────
rf2 = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    min_samples_split=10,
    max_depth=14,
    random_state=42
)
rf2.fit(X_train, y_train)

y_pred2 = rf2.predict(X_test)

print("\n" + "="*55)
print("TUNED MODEL (rf2) RESULTS")
print("="*55)
print(f"Training Accuracy: {rf2.score(X_train, y_train):.4f}")
print(f"Test Accuracy:     {accuracy_score(y_test, y_pred2):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred2, target_names=['Low Stress', 'High Stress']))

# Confusion matrix
cm2 = confusion_matrix(y_test, y_pred2)
plt.figure(figsize=(6, 4))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Low Stress', 'High Stress'],
            yticklabels=['Low Stress', 'High Stress'])
plt.title('Confusion Matrix — Tuned Model (rf2)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf2.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances — Tuned Model (rf2)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("\nTop 10 Features:")
print(feature_importances.head(10).to_string(index=False))
