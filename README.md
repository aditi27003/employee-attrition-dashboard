# 👥 Employee Attrition Dashboard

**🚀 [Try the live app here](https://aditi-employee-attrition.streamlit.app)**. No installation needed, it opens in your browser.

An HR analytics dashboard that asks **"Why do employees quit?"** and answers with data and
**organisational psychology**.

It uses the IBM HR Analytics Employee Attrition dataset (1,470 employees, 237 of whom left)
and has three parts:

1. **Overview**: headline attrition rate and the top factors linked to people leaving, each
   explained with a psychology theory (Herzberg's two-factor theory, Maslow, Equity Theory,
   the Job Demands-Resources model).
2. **Explore a factor**: pick any factor (overtime, job satisfaction, work-life balance, age,
   income, travel and more) and see the attrition rate for each group.
3. **Risk predictor**: describe an employee with sliders and a logistic regression model
   estimates their chance of leaving. On unseen employees it catches about 78% of the people
   who actually left.

## Key findings from the data
- **Overtime** triples attrition: 31% of those working overtime left, vs 10% of those who didn't.
- **Low job involvement**: 34% left, vs 9% with very high involvement.
- **Age 18–25**: 36% left, the highest of any age group.
- **Bad work-life balance**: 31% left.
- **Frequent business travel**: 25% left, vs 8% for non-travellers.

These are associations, not proof of cause.

## Tech used
- **Python**, **pandas**: data cleaning and analysis
- **scikit-learn**: logistic regression prediction model
- **Streamlit**: interactive web dashboard

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows you can also double-click `start_app.bat`.

## Files
- `hr_attrition.csv`: the dataset
- `analysis.py`: data preparation, factor ranking and the prediction model
- `app.py`: the dashboard

## Data source
IBM HR Analytics Employee Attrition & Performance, a fictional dataset created by IBM data scientists
([IBM/employee-attrition-aif360](https://github.com/IBM/employee-attrition-aif360)).
