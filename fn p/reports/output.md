
# AutoML Report

## Dataset Overview
- Rows: 1985
- Columns: 11
- Target Variable: Has_Hypertension

## Data Info
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1985 entries, 0 to 1984
Data columns (total 11 columns):
 #   Column            Non-Null Count  Dtype   
---  ------            --------------  -----   
 0   Age               1985 non-null   int64   
 1   Salt_Intake       1985 non-null   float64 
 2   Stress_Score      1985 non-null   int64   
 3   BP_History        1985 non-null   category
 4   Sleep_Duration    1985 non-null   float64 
 5   BMI               1985 non-null   float64 
 6   Medication        1985 non-null   category
 7   Family_History    1985 non-null   category
 8   Exercise_Level    1985 non-null   category
 9   Smoking_Status    1985 non-null   category
 10  Has_Hypertension  1985 non-null   category
dtypes: category(6), float64(3), int64(2)
memory usage: 90.3 KB


## Descriptive Statistics
### Numeric Columns
{
  "Age": {
    "count": 1985.0,
    "mean": 50.34105793450882,
    "std": 19.442041967242456,
    "min": 18.0,
    "25%": 34.0,
    "50%": 50.0,
    "75%": 67.0,
    "max": 84.0
  },
  "Salt_Intake": {
    "count": 1985.0,
    "mean": 8.531687657430732,
    "std": 1.9949074335722896,
    "min": 2.5,
    "25%": 7.2,
    "50%": 8.5,
    "75%": 9.9,
    "max": 16.4
  },
  "Stress_Score": {
    "count": 1985.0,
    "mean": 4.979345088161209,
    "std": 3.142303156496189,
    "min": 0.0,
    "25%": 2.0,
    "50%": 5.0,
    "75%": 8.0,
    "max": 10.0
  },
  "Sleep_Duration": {
    "count": 1985.0,
    "mean": 6.452241813602016,
    "std": 1.5422073433503702,
    "min": 1.5,
    "25%": 5.4,
    "50%": 6.5,
    "75%": 7.5,
    "max": 11.4
  },
  "BMI": {
    "count": 1985.0,
    "mean": 26.015314861460958,
    "std": 4.512856546331595,
    "min": 11.9,
    "25%": 23.0,
    "50%": 25.9,
    "75%": 29.1,
    "max": 41.9
  }
}

### Categorical Columns
{
  "BP_History": {
    "count": 1985,
    "unique": 3,
    "top": "Normal",
    "freq": 796
  },
  "Medication": {
    "count": 1985,
    "unique": 5,
    "top": "None",
    "freq": 799
  },
  "Family_History": {
    "count": 1985,
    "unique": 2,
    "top": "No",
    "freq": 1000
  },
  "Exercise_Level": {
    "count": 1985,
    "unique": 3,
    "top": "Low",
    "freq": 936
  },
  "Smoking_Status": {
    "count": 1985,
    "unique": 2,
    "top": "Non-Smoker",
    "freq": 1417
  },
  "Has_Hypertension": {
    "count": 1985,
    "unique": 2,
    "top": "Yes",
    "freq": 1032
  }
}

## Groupby Analysis (Has_Hypertension)
{
  "Age": {
    "No": 46.0797481636936,
    "Yes": 54.276162790697676,
    "Unknown": NaN
  },
  "Salt_Intake": {
    "No": 8.294228751311646,
    "Yes": 8.750968992248062,
    "Unknown": NaN
  },
  "Stress_Score": {
    "No": 4.368310598111227,
    "Yes": 5.5436046511627906,
    "Unknown": NaN
  },
  "Sleep_Duration": {
    "No": 6.644071353620147,
    "Yes": 6.275096899224806,
    "Unknown": NaN
  },
  "BMI": {
    "No": 25.334102833158447,
    "Yes": 26.64437984496124,
    "Unknown": NaN
  }
}

## Feature Importance
{
  "BP_History": 0.2919234453511023,
  "Age": 0.12235992545319799,
  "Stress_Score": 0.09815921723655158,
  "Sleep_Duration": 0.09737075162164348,
  "BMI": 0.09619498427353346,
  "Salt_Intake": 0.09510622144252978,
  "Family_History": 0.09285294189182469,
  "Smoking_Status": 0.07905386066163617,
  "Medication": 0.016750436704582736,
  "Exercise_Level": 0.010228215363397958
}

## Data Cleaning
- Handled missing values, duplicates, and unnecessary columns
- Final shape: (1985, 11)

## Exploratory Data Analysis
- Generated 3 figures (see UI)
- Visualizations: Countplot, scatter plot, boxplot, correlation heatmap, pairplot, feature importance

## Model Performance
- Task Type: Classification
- Metrics:
  - Accuracy: 0.9672544080604534
  - Classification Report: {'No': {'precision': 0.958974358974359, 'recall': 0.9739583333333334, 'f1-score': 0.9664082687338501, 'support': 192.0}, 'Yes': {'precision': 0.9752475247524752, 'recall': 0.9609756097560975, 'f1-score': 0.9680589680589681, 'support': 205.0}, 'accuracy': 0.9672544080604534, 'macro avg': {'precision': 0.9671109418634172, 'recall': 0.9674669715447155, 'f1-score': 0.9672336183964091, 'support': 397.0}, 'weighted avg': {'precision': 0.9673773790864844, 'recall': 0.9672544080604534, 'f1-score': 0.9672606449596667, 'support': 397.0}}
