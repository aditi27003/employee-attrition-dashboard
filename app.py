"""
app.py  -  Employee Attrition Dashboard (Streamlit web app).

Run with:  streamlit run app.py

Three tabs:
  1. Overview        - how many people left, and the top reasons linked to it
  2. Explore a factor - pick any factor and see its effect, with the psychology behind it
  3. Risk predictor  - describe an employee and get their chance of leaving
"""

import streamlit as st

from analysis import FACTORS, attrition_by, load_data, predict_one, top_drivers, train_model

st.set_page_config(page_title="Employee Attrition Dashboard", page_icon="👥", layout="wide")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(_df):
    return train_model(_df)


df_all = get_data()

# ---- Sidebar filter ----
st.sidebar.header("Filter")
departments = sorted(df_all["Department"].unique())
chosen = st.sidebar.multiselect("Department", departments, default=departments)
df = df_all[df_all["Department"].isin(chosen)]
st.sidebar.caption("Data: IBM HR Analytics Employee Attrition dataset (1,470 employees).")

st.title("👥 Employee Attrition Dashboard")
st.write("Why do employees quit? Combining **HR data** with **organisational psychology**.")

if df.empty:
    st.warning("Pick at least one department.")
    st.stop()

# ---- Headline numbers ----
c1, c2, c3 = st.columns(3)
c1.metric("Employees", f"{len(df):,}")
c2.metric("Left the company", f"{df['Left'].sum():,}")
c3.metric("Attrition rate", f"{df['Left'].mean() * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["📌 Overview", "🔍 Explore a factor", "🎯 Risk predictor"])

# ---- Tab 1: overview ----
with tab1:
    drivers = top_drivers(df)
    st.subheader("Top 5 factors linked to people leaving")
    for i, r in drivers.head(5).iterrows():
        st.markdown(f"**{i + 1}. {r['Factor']}**: {r['Highest-risk group']} vs "
                    f"{r['Lowest-risk group']}  \n_{r['Why it matters']}_")
    with st.expander("See all factors ranked"):
        st.dataframe(drivers, use_container_width=True, hide_index=True)
    st.caption("'Gap' = difference in attrition % between the highest- and lowest-risk group. "
               "These show links, not proof of cause.")

# ---- Tab 2: explore one factor ----
with tab2:
    names = {name: col for col, (name, _) in FACTORS.items()}
    pick = st.selectbox("Choose a factor", list(names))
    col = names[pick]
    rates = attrition_by(df, col)
    st.subheader(f"% of employees who left, by {pick.lower()}")
    st.bar_chart(rates)
    st.info(f"🧠 Psychology: {FACTORS[col][1]}")
    counts = df.groupby(col).agg(Employees=("Left", "size"), Left=("Left", "sum"))
    counts["Attrition %"] = (counts["Left"] / counts["Employees"] * 100).round(1)
    st.dataframe(counts, use_container_width=True)

# ---- Tab 3: risk predictor ----
with tab3:
    model, columns, accuracy, recall = get_model(df_all)
    st.subheader("How likely is this employee to leave?")
    st.caption(f"Simple logistic regression model trained on 1,470 employees. On employees it "
               f"hadn't seen, it caught {recall * 100:.0f}% of the people who actually left "
               f"(overall accuracy {accuracy * 100:.0f}%).")
    a, b, c = st.columns(3)
    with a:
        age = st.slider("Age", 18, 60, 30)
        income = st.slider("Monthly income (dataset units)", 1000, 20000, 5000, step=500)
        years = st.slider("Years at company", 0, 40, 3)
        promo = st.slider("Years since last promotion", 0, 15, 1)
    with b:
        overtime = st.radio("Works overtime?", ["No", "Yes"], horizontal=True)
        marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        travel = st.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance = st.slider("Distance from home (km)", 1, 30, 5)
    with c:
        job_sat = st.slider("Job satisfaction (1 = low, 4 = very high)", 1, 4, 3)
        env_sat = st.slider("Environment satisfaction (1-4)", 1, 4, 3)
        wlb = st.slider("Work-life balance (1 = bad, 4 = best)", 1, 4, 3)
        involve = st.slider("Job involvement (1-4)", 1, 4, 3)
        stock = st.slider("Stock option level (0-3)", 0, 3, 1)

    person = {"Age": age, "MonthlyIncome": income, "YearsAtCompany": years,
              "YearsSinceLastPromotion": promo, "DistanceFromHome": distance,
              "JobSatisfaction": job_sat, "EnvironmentSatisfaction": env_sat,
              "WorkLifeBalance": wlb, "JobInvolvement": involve, "StockOptionLevel": stock,
              "OverTime": overtime, "MaritalStatus": marital, "BusinessTravel": travel}
    risk = predict_one(model, columns, person) * 100

    if risk >= 60:
        st.error(f"🔴 High risk of leaving: {risk:.0f}%")
    elif risk >= 35:
        st.warning(f"🟡 Medium risk of leaving: {risk:.0f}%")
    else:
        st.success(f"🟢 Low risk of leaving: {risk:.0f}%")
    st.caption("Try switching overtime to 'Yes' or lowering job satisfaction and watch the risk change.")
