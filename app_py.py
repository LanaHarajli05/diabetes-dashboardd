import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

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

# Dashboard title
st.title("Exploring Diabetes Risk Across Demographics and Clinical Indicators")

with st.expander("Project Overview", expanded=True):
    st.markdown("""
    This dashboard is built on a synthetic healthcare dataset comprising over 100,000 anonymized patient records. The dataset includes key demographic variables (age group, gender), lifestyle indicators (smoking history), and clinical features such as Body Mass Index (BMI), HbA1c levels, blood glucose levels, hypertension status, and presence of heart disease.

    Leveraging interactive visual analytics, this dashboard explores the prevalence and distribution of diabetes across these variables. The goal is to help healthcare stakeholders identify risk factors, guide preventive strategies, and enable data-informed decisions to improve chronic disease management and population health outcomes.
    """)

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your filter selections.")
else:
    # All figure definitions
    age_group_chart = filtered_df.groupby("age_group")["diabetes"].mean().reset_index()
    fig1 = px.bar(age_group_chart, x="age_group", y="diabetes", height=300)

    gender_chart = filtered_df.groupby("gender")["diabetes"].mean().reset_index()
    fig2 = px.pie(gender_chart, values="diabetes", names="gender", height=300)

    smoke_chart = filtered_df.groupby("smoking_history")["diabetes"].mean().reset_index()
    fig3 = px.bar(smoke_chart, x="smoking_history", y="diabetes", height=300)

    fig4 = px.box(filtered_df, x="diabetes", y="HbA1c_level", height=300)

    conditions = [
        (filtered_df["hypertension"] == 1) & (filtered_df["heart_disease"] == 1),
        (filtered_df["hypertension"] == 1),
        (filtered_df["heart_disease"] == 1)
    ]
    choices = ["Both", "Hypertension Only", "Heart Disease Only"]
    filtered_df["comorbidity"] = np.select(conditions, choices, default="None")
    comorb_chart = filtered_df.groupby("comorbidity")["diabetes"].mean().reset_index()
    fig5 = px.bar(comorb_chart, x="comorbidity", y="diabetes", height=300)

    fig6 = px.violin(filtered_df, x="diabetes", y="bmi", box=True, points="outliers", height=300)
    fig7 = px.box(filtered_df, x="diabetes", y="blood_glucose_level", height=300)

    heatmap_data = filtered_df.groupby(["gender", "age_group"])["diabetes"].mean().reset_index()
    fig8 = px.density_heatmap(heatmap_data, x="gender", y="age_group", z="diabetes", color_continuous_scale="Reds", height=300)

    numeric_cols = ["HbA1c_level", "blood_glucose_level", "bmi", "age"]
    corr_matrix = filtered_df[numeric_cols].corr()
    fig9 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="Blues", height=300)

    heatmap_df = filtered_df.groupby(["age_group", "smoking_history"])["diabetes"].mean().reset_index()
    fig10 = px.density_heatmap(heatmap_df, x="smoking_history", y="age_group", z="diabetes", color_continuous_scale="Reds", height=300)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical Indicators", "Comorbidities", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Diabetes by Age Group")
            st.plotly_chart(fig1, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("We can notice that diabetes prevalence increases with age, especially for age groups 50–65 and 65+. This highlights the elderly as a high-risk group that needs targeted interventions.")
        with col2:
            st.subheader("Diabetes by Gender")
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The pie chart shows us the distribution of Diabetes among gender, we can notice that the % of Male with Diabetes is higher than % of female with diabetes.")

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Diabetes Rate by Smoking History")
            st.plotly_chart(fig3, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("Current and former smokers show higher rates of diabetes compared to never-smokers, supporting the evidence that smoking is a modifiable risk factor for diabetes.")
        with col4:
            st.subheader("HbA1c Levels by Diabetes Status")
            st.plotly_chart(fig4, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The boxplot shows a clear separation in HbA1c levels between diabetic and non-diabetic individuals. Diabetic patients tend to have higher and more variable HbA1c values, supporting the use of HbA1c as a reliable marker for diabetes diagnosis.")

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Diabetes Rate by Comorbidity Type")
            st.plotly_chart(fig5, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The diabetes rate is significantly higher among individuals with hypertension and heart disease. Those who have both comorbidities show the highest prevalence of diabetes, highlighting a strong link between cardiovascular comorbidities and diabetes risk.")
        with col6:
            st.subheader("BMI Distribution by Diabetes Status")
            st.plotly_chart(fig6, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The violin plot shows that individuals with diabetes tend to have a higher and more spread-out BMI distribution, reinforcing that excess body weight is a significant risk factor for developing diabetes.")

    with tab4:
        col7, col8 = st.columns(2)
        with col7:
            st.subheader("Blood Glucose Level by Diabetes Status")
            st.plotly_chart(fig7, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The boxplot clearly shows that individuals with diabetes tend to have higher and more variable blood glucose levels. This supports the clinical definition of diabetes as a condition characterized by elevated blood sugar.")
        with col8:
            st.subheader("Diabetes Rate by Gender and Age Group")
            st.plotly_chart(fig8, use_container_width=True)
            with st.expander("Interpretation"):
                st.markdown("The heatmap reveals a strong age-related trend in diabetes prevalence, with older adults (65+) having the highest diabetes rates. Female=0.18 and Male=0.22.")

        st.subheader("Correlation Between Clinical Indicators")
        st.plotly_chart(fig9, use_container_width=True)
        with st.expander("Interpretation"):
            st.markdown("The correlation matrix shows that age and BMI have the strongest relationship (r=0.337), followed by a modest correlation between glucose and HbA1c (r=0.1668). This suggests a need for multifactorial tests rather than relying on a single indicator like weight.")

        st.subheader("Diabetes Rate by Age Group and Smoking History")
        st.plotly_chart(fig10, use_container_width=True)
        with st.expander("Interpretation"):
            st.markdown("The heatmap shows a compounded effect of age and smoking on diabetes prevalence. Older individuals, especially those aged 65+, reveal significantly higher diabetes rates. Even in middle age (50-65), smoking history is associated with increased diabetes prevalence.")

# Footer
st.markdown("---")
st.markdown("*\u201cNever be ashamed of being diabetic. It\u2019s not a weakness; it\u2019s a story of strength and resilience.\u201d*")
st.markdown("*Developed by Lana Harajli*")

# Reduce padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


      





   




