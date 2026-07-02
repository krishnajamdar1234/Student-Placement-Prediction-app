import streamlit as st
import pickle as pkl 
import numpy as np 

st.set_page_config(page_title =  "Student Placement Prediction" , page_icon = "🎓" , layout = "centered")

# Load Model
model = pkl.load(open("placement_model.pkl", "rb"))

# Custom CSS
st.markdown("""
<style>
.main{
    background-color:#f5f7fa;
}
.stButton>button{
    width:100%;
    background-color:#0d6efd;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}
.stButton>button:hover{
    background-color:#084298;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Student Placement Prediction System")
st.write("Predict student placement chances using Machine Learning.")

st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:
    studid = st.number_input("Student ID" , 0,100)
    cgpa = st.number_input("CGPA", 0.0, 10.0)
    internships = st.number_input("Internships", 0, 20)
    projects = st.number_input("Projects", 0, 20)
    aptitude = st.number_input("Aptitude Score", 0, 100)
    softskills = st.number_input("Soft Skills Rating", 0, 10)

with col2:
    workshops = st.number_input("Workshops/Certifications", 0, 20)
    extra = st.selectbox("Extracurricular Activities", ["No","Yes"])
    training = st.selectbox("Placement Training", ["No","Yes"])
    ssc = st.number_input("SSC Marks", 0, 100)
    hsc = st.number_input("HSC Marks", 0, 100)

    # Encoding
extra = 1 if extra=="Yes" else 0
training = 1 if training=="Yes" else 0

# Prediction
if st.button("🔍 Predict Placement"):

    data = np.array([[cgpa, internships, projects,
                      workshops, aptitude,
                      softskills, extra,
                      training, ssc, hsc]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("🎉 Congratulations! High Chances of Placement.")
        st.balloons()

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data)[0][1] * 100
            st.info(f"Placement Probability : {prob:.2f}%")

    else:
        st.error("❌ Placement Chances are Low.")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data)[0][1] * 100
            st.warning(f"Placement Probability : {prob:.2f}%")

st.divider()

st.caption("Developed by Krishna Jamdar | Student Placement Prediction Project")