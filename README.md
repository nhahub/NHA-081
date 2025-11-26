# 🫀 Healthcare Project: Indicators of Heart Disease (2022 Update)

## 📊 1. Dataset Description

This project uses the **“Indicators of Heart Disease (2022 Update)”** dataset.  
The dataset contains health-related information collected from U.S. adults to identify factors that may increase the risk of heart disease.

- **Number of Rows:** 445,132  
- **Number of Columns:** 40  
- **Source:** [Kaggle - Indicators of Heart Disease (2022)](https://www.kaggle.com)  
- **Objective:** Analyze the dataset, clean and preprocess it, then prepare it for data visualization and predictive modeling.

### 🧾 Feature Overview

| Feature                          | Description |
|-----------------------------------|-------------|
| `State`                           | U.S. state of residence |
| `Sex`                             | Biological sex of the participant |
| `GeneralHealth`                   | Self-rated general health |
| `PhysicalHealthDays`              | Number of days physical health was not good |
| `MentalHealthDays`                | Number of days mental health was not good |
| `LastCheckupTime`                 | Time since last routine checkup |
| `PhysicalActivities`              | Whether the participant engaged in physical activities |
| `SleepHours`                      | Average hours of sleep per night |
| `RemovedTeeth`                    | Number of permanent teeth removed |
| `HadHeartAttack`                  | History of heart attack (Yes/No) |
| `HadAngina`                       | History of angina or coronary heart disease |
| `HadStroke`                       | History of stroke |
| `HadAsthma`                       | History of asthma |
| `HadSkinCancer`                   | History of skin cancer |
| `HadCOPD`                         | History of chronic obstructive pulmonary disease |
| `HadDepressiveDisorder`           | History of depressive disorder |
| `HadKidneyDisease`               | History of kidney disease |
| `HadArthritis`                   | History of arthritis |
| `HadDiabetes`                    | History of diabetes |
| `DeafOrHardOfHearing`            | Hearing difficulties |
| `BlindOrVisionDifficulty`       | Vision difficulties |
| `DifficultyConcentrating`       | Difficulty concentrating or remembering |
| `DifficultyWalking`             | Difficulty walking or climbing stairs |
| `DifficultyDressingBathing`     | Difficulty dressing or bathing |
| `DifficultyErrands`            | Difficulty doing errands alone |
| `SmokerStatus`                 | Smoking status |
| `ECigaretteUsage`              | E-cigarette usage |
| `ChestScan`                    | History of chest CT scan |
| `RaceEthnicityCategory`        | Race and ethnicity |
| `AgeCategory`                  | Age group |
| `HeightInMeters`              | Height in meters |
| `WeightInKilograms`           | Weight in kilograms |
| `BMI`                          | Body Mass Index |
| `AlcoholDrinkers`             | Alcohol consumption |
| `HIVTesting`                  | HIV testing status |
| `FluVaxLast12`                | Flu vaccination in the last 12 months |
| `PneumoVaxEver`               | Pneumococcal vaccination |
| `TetanusLast10Tdap`           | Tetanus vaccination in the last 10 years |
| `HighRiskLastYear`           | High risk for severe illness in the last year |
| `CovidPos`                    | COVID-19 positivity status |

---

## 🧹 2. Step 1: Data Cleaning

Data cleaning ensures data quality, consistency, and reliability for analysis.  
The following steps were applied:

- **Handling Missing Values:**  
  - Identified missing values across all columns.  
  - Applied appropriate strategies (drop or impute) depending on data importance.

- **Text Standardization:**  
  - Unified categorical values (e.g., `Yes` / `No`) and removed extra spaces.  
  - Normalized text casing.

- **Duplicate Removal:**  
  - Checked for duplicate records and removed them to avoid redundancy.

- **Data Type Conversion:**  
  - Converted numeric columns like `HeightInMeters`, `WeightInKilograms`, `BMI`, and health day counts to numeric types.  
  - Converted categorical variables to `category` type for optimization.

- **Inconsistency Check:**  
  - Fixed inconsistent values and ensured standardized labels across categorical fields.

---

## 📈 3. Step 2: Data Visualization & Preprocessing

### 🖼️ Data Visualization

To understand the data distribution and detect hidden patterns:

- Plotted the **distribution of heart-related conditions** (`HadHeartAttack`, `HadAngina`, etc.).  
- Explored relationships between **BMI**, **AgeCategory**, and **Sex** with heart disease indicators.  
- Visualized **vaccination**, **smoking**, and **alcohol habits** impact on health indicators.  
- Detected outliers in numerical features (`BMI`, `SleepHours`, `PhysicalHealthDays`) using boxplots.  
- Checked data balance for key target variables.

### ⚙️ Data Preprocessing

Prepared the dataset for modeling:

- **Encoding:**  
  - Transformed categorical columns into numeric format using One-Hot Encoding or Label Encoding.

- **Scaling:**  
  - Standardized numerical columns to improve model performance.

- **Balancing:**  
  - Addressed class imbalance for target features like `HadHeartAttack` if necessary.

- **Splitting:**  
  - Divided data into training and testing sets for future predictive modeling.

--

## 🛠️ Tools & Libraries

- [Python](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  
- [Scikit-learn](https://scikit-learn.org/)

--


## 📌 Team
**Ahmed Sameh**  
**AbdalluH Ahmed**  
**Mohamed Yasser**  
**Rewan Adel**  


