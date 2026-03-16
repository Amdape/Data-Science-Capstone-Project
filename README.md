# 📱 Analysis of the Effect of Screen Time on Stress Level

> **TS Academy Data Science Capstone Project — Group 5**
> **Track:** Supervised Learning — Binary Classification
> **Dataset:** Smartphone Usage & Productivity Dataset (50,000 records) — Kaggle

---

## 👥 Group Members

| # | Name | Email |
|---|------|-------|
| 1 | Ejiayelia Festus Idiake | fidefide098@gmail.com |
| 2 | Idris Adeshina Sulaimon | Sulaimonidris61@gmail.com |
| 3 | Amao Daniel Pelumi | amaodanielpelumi156@gmail.com |
| 4 | Monday Thankgod Namson | tamytrust@gmail.com |
| 5 | Sherif Olalekan Salami | salammyolalekan@gmail.com |
| 6 | Shaibu Atekojo Wisdom | shaibuateko@gmail.com |
| 7 | Anyanwu Chizoba Mercy | chizobamercya19@gmail.com |
| 8 | Adebayo David | uniquetristan90@gmail.com |
| 9 | Okolo Gideon | gideonokolo2018@gmail.com |
| 10 | Abiodun Toheeb Akande | Herbey604@gmail.com |
| 11 | Ezeh Daniel Chekwube | faradayezeh@gmail.com |

---

## 📌 Project Overview

Smartphone usage has become central to modern life, yet its psychological costs — particularly elevated stress — are not fully understood. This project investigates whether smartphone usage habits, lifestyle factors, and demographic variables can predict whether an individual experiences **High or Low stress**.

We sourced a 50,000-record dataset from Kaggle, cleaned and explored it thoroughly, and built a **Random Forest binary classifier** to predict stress levels. The project follows the full data science workflow: data cleaning → EDA → feature engineering → modelling → evaluation.

---

## 🗂️ Repository Structure

```
├── Group5_Corrected_Notebook.py     # Full Python analysis script (run in Colab)
├── Group5_Capstone_FINAL.docx       # Final written report
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 📊 Dataset

- **Source:** [Kaggle — Smartphone Usage and Productivity Dataset](https://www.kaggle.com/datasets)
- **Raw size:** 50,000 records × 13 columns
- **After cleaning:** 35,350 records (29.3% removed due to logical inconsistencies)
- **Target variable:** `Stress_Level` (1–10) → binarised to `Stress_Binary` (0 = Low, 1 = High)

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Average_Daily_Phone_Hours | Numerical | Daily phone screen time (hours) |
| Social_Media_Hours | Numerical | Daily social media usage (hours) |
| Work_Productivity_Score | Numerical | Self-reported productivity score |
| Sleep_Hours | Numerical | Average nightly sleep (hours) |
| App_Usage_Count | Numerical | Number of apps used |
| Caffeine_Intake_Cups | Numerical | Daily caffeine consumption |
| Weekend_Screen_Time_Hours | Numerical | Weekend screen time (hours) |
| Total_Weekly_Screen_Time | Engineered | (Social_Media_Hours × 5) + Weekend_Screen_Time_Hours |
| Age | Numerical | Respondent age (18–60) |
| Gender | Categorical | Male / Female / Other |
| Occupation | Categorical | Business Owner / Freelancer / Student / Professional |
| Device_Type | Categorical | Primary smartphone device |

---

## ⚙️ How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Group5_Corrected_Notebook.py` or copy its contents into a new notebook
3. Upload the dataset CSV to your Colab session:
   ```python
   # Update this path to match where you uploaded the CSV
   df_smartphone = pd.read_csv('/content/Smartphone_Usage_Productivity_Dataset_50000 (1).csv')
   ```
4. Run all cells

### Option 2 — Local Machine

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Run the script
python Group5_Corrected_Notebook.py
```

---

## 🔑 Key Results

### Data Cleaning
| Check | Result |
|-------|--------|
| Missing values | 0 |
| Duplicates | 0 |
| Logical inconsistencies removed | 14,650 rows (29.3%) |
| Clean records | 35,350 |

### Model Performance

| Model | Train Accuracy | Test Accuracy | Macro F1 |
|-------|---------------|---------------|----------|
| Baseline rf1 (default params) | 100.0% | 58.0% | 0.58 |
| Tuned rf2 (optimised params) | 85.9% | **59.1%** | **0.59** |

> ✅ The tuned model (rf2) outperforms the baseline across all metrics with significantly reduced overfitting.

### Top 5 Most Important Features (rf2)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Average_Daily_Phone_Hours | 0.148 |
| 2 | Social_Media_Hours | 0.121 |
| 3 | Total_Weekly_Screen_Time | 0.119 |
| 4 | Work_Productivity_Score | 0.114 |
| 5 | Sleep_Hours | 0.113 |

> 📌 Screen time variables dominate the top 3 features, accounting for **38.8%** of total predictive importance.

### Stress by Demographic Group

| Group | Category | Mean Stress |
|-------|----------|-------------|
| Gender | Other | 5.44 |
| Gender | Male | 5.41 |
| Gender | Female | 5.39 |
| Occupation | Student | 5.43 |
| Occupation | Professional | 5.38 |
| Age Group | Under 20 | 5.44 |
| Age Group | 21–30 | 5.36 |

---

## 🧠 Key Findings

- **Screen time is the strongest predictor of stress.** Daily phone hours, social media usage, and weekly screen time together account for nearly 39% of the model's predictive power.
- **Sleep and productivity matter equally.** Lower sleep and lower productivity are strongly associated with higher stress — stress is not driven by screen time alone.
- **Students and under-20s show marginally higher stress** than other groups, consistent with social media pressure and academic demands.
- **Demographic differences are small** — all group mean stress scores fall within a 0.08-point range on a 10-point scale.

---

## 🛠️ Technologies Used

- **Python 3** (Google Colab)
- `pandas` — data manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualisation
- `scikit-learn` — machine learning (RandomForestClassifier, train_test_split, classification_report)

---

## 📄 Report

The full written report (`Group5_Capstone_FINAL.docx`) covers:
1. Introduction & Problem Statement
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering & Encoding
5. Modelling & Results
6. Feature Importance Analysis
7. Conclusions
8. Recommendations

---

## 🏫 Acknowledgement

This project was completed as part of the **TS Academy Data Science Programme**, Capstone Project module. All group members contributed collaboratively to the analysis, modelling, and reporting.

---

*March 2026 — Group 5*
