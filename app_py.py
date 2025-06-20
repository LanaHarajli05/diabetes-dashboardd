import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np  # Required for comorbidity logic


# Page configuration
st.set_page_config(page_title="Exploring Diabetes Risk Across Demographics and Clinical Indicators", layout="wide")

# Load data
df = pd.read_csv("diabetes_clean.csv")

# Sidebar filters
st.sidebar.title("Filters")
gender_filter = st.sidebar.multiselect("Select Gender", options=df["gender"].unique(), default=df["gender"].unique())
age_group_filter = st.sidebar.multiselect("Select Age Group", options=df["age_group"].unique(), default=df["age_group"].unique())
smoking_filter = st.sidebar.multiselect("Select Smoking History", options=df["smoking_history"].unique(), default=df["smoking_history"].unique())
heart_disease_filter = st.sidebar.multiselect("Select Heart Disease", options=df["heart_disease"].unique(), default=df["heart_disease"].unique())
hypertension_filter = st.sidebar.multiselect("Select Hypertension", options=df["hypertension"].unique(), default=df["hypertension"].unique())

# Filter data
filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["age_group"].isin(age_group_filter)) &
    (df["smoking_history"].isin(smoking_filter)) &
    (df["heart_disease"].isin(heart_disease_filter)) &
    (df["hypertension"].isin(hypertension_filter))
]

# Dashboard title and intro
st.title("Exploring Diabetes Risk Across Demographics and Clinical Indicators")
st.markdown("""
This dashboard is built on a synthetic healthcare dataset comprising over 100,000 anonymized patient records. The dataset includes key demographic variables (age group, gender), lifestyle indicators (smoking history), and clinical features such as Body Mass Index (BMI), HbA1c levels, blood glucose levels, hypertension status, and presence of heart disease.

Leveraging interactive visual analytics, this dashboard explores the prevalence and distribution of diabetes across these variables. The goal is to help healthcare stakeholders identify risk factors, guide preventive strategies, and enable data-informed decisions to improve chronic disease management and population health outcomes.
""")

# NOTE: All indentation errors fixed below (inside expanders)

# Row 1
col1, col2 = st.columns(2)
with col1:
    st.subheader("Diabetes Prevalence by Age Group")
    age_group_chart = filtered_df.groupby("age_group")["diabetes"].mean().reset_index()
    fig1 = px.bar(age_group_chart, x="age_group", y="diabetes")
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""We can notice that diabetes prevalence increases with age, especially for age groups 50–65 and 65+. This highlights the elderly as a high-risk group that needs targeted interventions.""")

with col2:
    st.subheader("Diabetes Rates by Gender")
    gender_chart = filtered_df.groupby("gender")["diabetes"].mean().reset_index()
    fig2 = px.pie(
        gender_chart,
        values="diabetes",
        names="gender",
        title="Diabetes Rate by Gender"
    )
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""The pie chart shows the distribution of diabetes among gender. We can notice that the percentage of males with diabetes is higher than that of females.""")



# Row 2
col3, col4 = st.columns(2)
with col3:
    st.subheader("Diabetes Rate by Smoking History")
    smoke_chart = filtered_df.groupby("smoking_history")["diabetes"].mean().reset_index()
    fig3 = px.bar(smoke_chart, x="smoking_history", y="diabetes")
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""Current and former smokers show higher rates of diabetes compared to never-smokers, supporting the evidence that smoking is a modifiable risk factor for diabetes.""")

with col4:
    st.subheader("HbA1c Levels by Diabetes Status")
    fig4 = px.box(filtered_df, x="diabetes", y="HbA1c_level")
    st.plotly_chart(fig4, use_container_width=True)
    with st.expander("Interpretation"):
        st.markdown("""The boxplot shows a clear separation in HbA1c levels between diabetic and non-diabetic individuals. Diabetic patients tend to have higher and more variable HbA1c values, supporting the use of HbA1c as a reliable marker for diabetes diagnosis.""")

# Row 3: Comorbidity chart
st.markdown("---")
st.subheader("Diabetes Rate by Comorbidity Type")

conditions = [
    (filtered_df["hypertension"] == 1) & (filtered_df["heart_disease"] == 1),
    (filtered_df["hypertension"] == 1),
    (filtered_df["heart_disease"] == 1)
]
choices = ["Both", "Hypertension Only", "Heart Disease Only"]
filtered_df["comorbidity"] = np.select(conditions, choices, default="None")

comorb_chart = filtered_df.groupby("comorbidity")["diabetes"].mean().reset_index()
fig5 = px.bar(comorb_chart, x="comorbidity", y="diabetes")
st.plotly_chart(fig5, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The diabetes rate is significantly higher among individuals with hypertension and heart disease. Those who have both comorbidities show the highest prevalence of diabetes, highlighting a strong link between cardiovascular comorbidities and diabetes risk.""")

# Row 4: BMI violin
st.markdown("---")
st.subheader("BMI Distribution by Diabetes Status")
fig6 = px.violin(filtered_df, x="diabetes", y="bmi", box=True, points="outliers")
st.plotly_chart(fig6, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The violin plot shows that individuals with diabetes tend to have a higher and more spread-out BMI distribution, reinforcing that excess body weight is a significant risk factor for developing diabetes.""")

# Row 5: Blood glucose
st.markdown("---")
st.subheader("Blood Glucose Level by Diabetes Status")
fig7 = px.box(filtered_df, x="diabetes", y="blood_glucose_level")
st.plotly_chart(fig7, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The boxplot clearly shows that individuals with diabetes tend to have higher and more variable blood glucose levels. This supports the clinical definition of diabetes as a condition characterized by elevated blood sugar.""")

# Row 6: Heatmap by Gender and Age Group
st.markdown("---")
st.subheader("Diabetes Rate by Gender and Age Group")
heatmap_data = filtered_df.groupby(["gender", "age_group"])["diabetes"].mean().reset_index()
fig8 = px.density_heatmap(heatmap_data, x="gender", y="age_group", z="diabetes", color_continuous_scale="Reds")
st.plotly_chart(fig8, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The heatmap reveals a strong age-related trend in diabetes prevalence, with older adults (65+) having the highest diabetes rates. Female=0.18 and Male=0.22.""")

# Row 7: Correlation Matrix
st.markdown("---")
st.subheader("Correlation Between Clinical Indicators")
numeric_cols = ["HbA1c_level", "blood_glucose_level", "bmi", "age"]
corr_matrix = filtered_df[numeric_cols].corr()
fig9 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Blues")
st.plotly_chart(fig9, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The correlation matrix shows that age and BMI have the strongest relationship (r=0.337), followed by a modest correlation between glucose and HbA1c (r=0.1668). This suggests a need for multifactorial tests rather than relying on a single indicator like weight.""")

# Row 8: Age Group & Smoking History Heatmap
st.markdown("---")
st.subheader("Diabetes Rate by Age Group and Smoking History")
heatmap_df = filtered_df.groupby(["age_group", "smoking_history"])["diabetes"].mean().reset_index()
fig10 = px.density_heatmap(heatmap_df, x="smoking_history", y="age_group", z="diabetes", color_continuous_scale="Reds")
st.plotly_chart(fig10, use_container_width=True)
with st.expander("Interpretation"):
    st.markdown("""The heatmap shows a compounded effect of age and smoking on diabetes prevalence. Older individuals, especially those aged 65+, reveal significantly higher diabetes rates. Even in middle age (50-65), smoking history is associated with increased diabetes prevalence.""")

# Footer
st.markdown("---")
st.markdown("*\u201cNever be ashamed of being diabetic. It\u2019s not a weakness; it\u2019s a story of strength and resilience.\u201d*")
st.markdown("*Developed by Lana Harajli*")

# Optional: reduce padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)



    




   




