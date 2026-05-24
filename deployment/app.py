DurationOfPitch = st.number_input(
    "Duration Of Pitch",
    min_value=1,
    max_value=120,
    value=20
)

# ==========================================
# Prepare Input Data
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
# Prediction
# ==========================================

if st.button("Predict Purchase"):

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "Customer is likely to purchase the Wellness Tourism Package."
            )
        else:
            st.error(
                "Customer is unlikely to purchase the Wellness Tourism Package."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
