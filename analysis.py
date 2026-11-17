"""
analysis.py  -  the "brain" of the attrition dashboard.

Data: IBM HR Analytics Employee Attrition dataset (1,470 employees, fictional
but realistic, created by IBM data scientists for learning).
'Attrition' = Yes means the employee left the company.

This file:
  1. Loads the data and adds easy-to-read columns (age groups, labels).
  2. Finds which factors are linked to people leaving.
  3. Trains a small prediction model ("how likely is this person to leave?").
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import recall_score

LEVELS = {1: "1 Low", 2: "2 Medium", 3: "3 High", 4: "4 Very High"}
BALANCE = {1: "1 Bad", 2: "2 Good", 3: "3 Better", 4: "4 Best"}

# Factors we compare, with a plain-English name and a psychology note.
FACTORS = {
    "OverTime": ("Works overtime", "Long hours drain energy and lead to burnout, a major reason people quit."),
    "JobSatisfactionLabel": ("Job satisfaction", "People who don't enjoy their work start looking elsewhere (Herzberg's motivators)."),
    "EnvironmentSatisfactionLabel": ("Workplace environment satisfaction", "A poor physical or social environment is a 'hygiene factor': it doesn't motivate, but its absence pushes people out."),
    "WorkLifeBalanceLabel": ("Work-life balance", "Without recovery time, stress builds up (Job Demands-Resources model)."),
    "JobInvolvementLabel": ("Job involvement", "Low involvement means low engagement, an early warning sign of quitting."),
    "RelationshipSatisfactionLabel": ("Relationship satisfaction (with colleagues)", "Belonging is a basic human need (Maslow). Weak work relationships make leaving easier."),
    "MaritalStatus": ("Marital status", "Single employees often have fewer ties keeping them in one place."),
    "BusinessTravel": ("Business travel", "Frequent travel disrupts routine and family life."),
    "AgeGroup": ("Age group", "Younger employees are still exploring careers and switch jobs more often."),
    "IncomeGroup": ("Monthly income group", "Feeling under-paid hurts perceived fairness (Equity Theory)."),
    "PromotionGap": ("Years since last promotion", "Feeling 'stuck' reduces motivation and growth expectations."),
    "Department": ("Department", "Different teams have different workloads, pressures and cultures."),
    "StockOptionLevel": ("Stock option level", "Shares in the company give employees a reason to stay."),
}


def load_data(path: str = "hr_attrition.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Left"] = (df["Attrition"] == "Yes").astype(int)
    df["JobSatisfactionLabel"] = df["JobSatisfaction"].map(LEVELS)
    df["EnvironmentSatisfactionLabel"] = df["EnvironmentSatisfaction"].map(LEVELS)
    df["JobInvolvementLabel"] = df["JobInvolvement"].map(LEVELS)
    df["RelationshipSatisfactionLabel"] = df["RelationshipSatisfaction"].map(LEVELS)
    df["WorkLifeBalanceLabel"] = df["WorkLifeBalance"].map(BALANCE)
    df["AgeGroup"] = pd.cut(df["Age"], [17, 25, 35, 45, 60],
                            labels=["18-25", "26-35", "36-45", "46-60"]).astype(str)
    df["IncomeGroup"] = pd.qcut(df["MonthlyIncome"], 4,
                                labels=["1 Lowest 25%", "2 Lower-middle", "3 Upper-middle", "4 Highest 25%"]).astype(str)
    df["PromotionGap"] = pd.cut(df["YearsSinceLastPromotion"], [-1, 0, 2, 5, 20],
                                labels=["0 This year", "1-2 years", "3-5 years", "6+ years"]).astype(str)
    return df


def attrition_by(df: pd.DataFrame, factor: str) -> pd.Series:
    """% of people who left, for each group of a factor."""
    return (df.groupby(factor)["Left"].mean() * 100).sort_index()


def top_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Rank factors by the gap between their worst and best group.
    A big gap means the factor is strongly linked to people leaving."""
    rows = []
    for col, (name, note) in FACTORS.items():
        rates = attrition_by(df, col)
        rates = rates[df.groupby(col).size() >= 20]       # ignore tiny groups
        if len(rates) < 2:
            continue
        rows.append({
            "Factor": name,
            "Highest-risk group": f"{rates.idxmax()} ({rates.max():.0f}% left)",
            "Lowest-risk group": f"{rates.idxmin()} ({rates.min():.0f}% left)",
            "Gap (points)": round(rates.max() - rates.min(), 1),
            "Why it matters": note,
        })
    return pd.DataFrame(rows).sort_values("Gap (points)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction model
# ---------------------------------------------------------------------------
MODEL_NUMBERS = ["Age", "MonthlyIncome", "YearsAtCompany", "YearsSinceLastPromotion",
                 "DistanceFromHome", "JobSatisfaction", "EnvironmentSatisfaction",
                 "WorkLifeBalance", "JobInvolvement", "StockOptionLevel"]
MODEL_CATEGORIES = ["OverTime", "MaritalStatus", "BusinessTravel"]


def to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns into 0/1 columns the model can read."""
    X = pd.get_dummies(df[MODEL_NUMBERS + MODEL_CATEGORIES], columns=MODEL_CATEGORIES, dtype=int)
    return X


def train_model(df: pd.DataFrame):
    """Logistic regression: a simple, explainable model that outputs a probability.
    Returns (model, column list, accuracy, recall).
    Recall = out of the people who really left, how many the model flagged."""
    X = to_features(df)
    y = df["Left"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    recall = recall_score(y_test, model.predict(X_test))
    return model, list(X.columns), accuracy, recall


def predict_one(model, columns: list, person: dict) -> float:
    """Probability (0-1) that one described employee leaves."""
    X = to_features(pd.DataFrame([person])).reindex(columns=columns, fill_value=0)
    return float(model.predict_proba(X)[0, 1])
