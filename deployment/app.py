
import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# ==========================================
# Download and Load Trained Model
# ==========================================

model_path = hf_hub_download(
    repo_id="RahulGolande/tourism-package-prediction-model",
    filename="best_tourism_model.pkl"
)

model = joblib.load(model_path)

# ==========================================
# Streamlit UI
# ==========================================

st.title("Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase
the Wellness Tourism Package.
""")

# ==========================================
# User Inputs
# ==========================================

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Free Lancer", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

PreferredPropertyStar = st.slider(
    "Preferred Property Star",
    min_value=1,
    max_value=5,
    value=3
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=20,
    value=2
)

Passport = st.selectbox(
    "Passport",
    [0, 1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=1000000,
    value=50000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=120,
    value=15
)

# ==========================================
# Create Input DataFrame
# ==========================================

input_data = pd.DataFrame([{

    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch

}])

# ==========================================
# Prediction Button
# ==========================================

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the "
            "Wellness Tourism Package."
        )

    else:

        st.error(
            "Customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )
