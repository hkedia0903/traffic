# Import necessary libraries
from mapie.regression import MapieRegressor 
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import warnings 
warnings.filterwarnings("ignore")


dt_pickle = open('traffic.pickle', 'rb') 
clf = pickle.load(dt_pickle) 
dt_pickle.close()

st.title('Traffic Volume Predictor') 
st.write("Utilize our advanced Machine Learning application to predict traffic volume.")

st.image('traffic_sidebar.jpg', width = 400)

st.sidebar.image('traffic_sidebar.jpg', caption="Traffic Volume Predictor")
st.sidebar.subheader("Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

expected_cols = ['temp','rain_1h','snow_1h','clouds_all','hour','holiday_Christmas Day','holiday_Columbus Day','holiday_Independence Day','holiday_Labor Day','holiday_Martin Luther King Jr Day','holiday_Memorial Day',
    'holiday_New Years Day','holiday_None','holiday_State Fair','holiday_Thanksgiving Day','holiday_Veterans Day','holiday_Washingtons Birthday','weather_main_Clear','weather_main_Clouds','weather_main_Drizzle','weather_main_Fog','weather_main_Haze','weather_main_Mist',
    'weather_main_Rain','weather_main_Smoke','weather_main_Snow','weather_main_Squall','weather_main_Thunderstorm','month_April','month_August','month_December','month_February','month_January','month_July','month_June','month_March','month_May',
    'month_November','month_October','month_September', 'weekday_Friday','weekday_Monday','weekday_Saturday','weekday_Sunday', 'weekday_Thursday','weekday_Tuesday','weekday_Wednesday']


# option 1:
with st.sidebar.expander("Option 1: Upload CSV file", expanded=False):
    st.write("Upload a CSV file containing the traffic details.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) 

    st.write("### Sample Data Format for Upload")
    sample_df = pd.read_csv("Traffic_Volume.csv") 

    sample_df['date_time'] = pd.to_datetime(sample_df['date_time'], format='%m/%d/%y %H:%M')
    sample_df['month'] = sample_df['date_time'].dt.month_name()
    sample_df['weekday'] = sample_df['date_time'].dt.day_name()
    sample_df['hour'] = sample_df['date_time'].dt.hour
    sample_df = sample_df.drop(columns=['date_time', 'traffic_volume']) 
    # used chatgpt 5.0 to help with this date_time stuff 

    st.dataframe(sample_df.head())
    st.warning("Ensure your uploaded file has the same column names and data types as shown above.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X = pd.get_dummies(df, columns=['holiday', 'weather_main', 'month', 'weekday'], drop_first=False)

    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]  

    st.session_state["uploaded_df"] = df
    st.session_state["X_uploaded"] = X


# option 2
with st.sidebar.expander("Option 2: Fill Out Form", expanded=False):
    st.write("Enter the traffic details manually using the form below.")

    holiday_input = st.selectbox("Holiday", ["None", 'Columbus Day', 'Veterans Day', 'Thanksgiving Day', 'Christmas Day',
                                             'New Years Day', 'Washingtons Birthday', 'Memorial Day', 'Independence Day',
                                             'State Fair', 'Labor Day', 'Martin Luther King Jr Day'])
    temperature = st.number_input("Average temperature in Kelvin", value=281.21)
    rain = st.number_input("Amount in mm of rain that occurred in the hour", value=0.33)
    snow = st.number_input("Amount in mm of snow that occurred in the hour", value=0.00)
    cloud = st.number_input("Percentage of cloud cover", min_value=0, max_value=100, value=49)
    weather_input = st.selectbox("Choose the current weather", ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Thunderstorm',
                                                                'Snow', 'Squall', 'Smoke'])
    month_input = st.selectbox("Choose month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                                'September', 'October', 'November', 'December'])
    day_input = st.selectbox("Choose day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    hour_input = st.selectbox("Hour of the day (0-23)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], index=0)

    input_df = pd.DataFrame({
        'temp': [temperature],
        'rain_1h': [rain],
        'snow_1h': [snow],
        'clouds_all': [cloud],
        'month': [month_input],
        'weekday': [day_input],
        'hour': [hour_input],
        'holiday': [holiday_input],
        'weather_main': [weather_input]
    })

    input_df = pd.get_dummies(input_df, columns=['holiday', 'weather_main'], drop_first=False)

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    if st.button("Submit Form Data"):
        st.session_state["form_submitted"] = True
        st.session_state["input_features"] = input_df

if "input_features" not in st.session_state:
    st.session_state["input_features"] = None

if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

with open('traffic.pickle', 'rb') as f:
    mapie_model = pickle.load(f)

if uploaded_file is not None:
    st.success("CSV file uploaded successfully!")
elif st.session_state.get("form_submitted", False):
    st.success("Form data submitted successfully!")
else:
    st.info("Please choose a data input method to proceed.")

alpha = st.slider("Select alpha for prediction interval", min_value=0.01, max_value=0.5, value=0.1)


if not st.session_state["form_submitted"] and uploaded_file is None:
    st.subheader("Predicting Traffic Volume...")
    st.metric(label="Predicted Traffic Volume", value="0")
    st.markdown(f"**Prediction Interval** ({(1-alpha)*100:.1f}%): [0, 664]")

if st.session_state["form_submitted"] and st.session_state["input_features"] is not None:
    X_input = st.session_state["input_features"]
    y_pred, y_pis = mapie_model.predict(X_input, alpha=alpha)

    prediction = float(y_pred[0])
    lower_bound = float(y_pis[0, 0])
    upper_bound = float(y_pis[0, 1])

    st.subheader("Predicting Traffic Volume...")
    st.metric(label="Predicted Traffic Volume", value=f"{prediction:,.0f}")
    st.markdown(f"**Prediction Interval** ({(1-alpha)*100:.1f}%): [{lower_bound:,.0f}, {upper_bound:,.0f}]")

elif uploaded_file is not None:

    df = st.session_state["uploaded_df"]
    X = st.session_state["X_uploaded"]

    y_pred, y_pis = mapie_model.predict(X, alpha=alpha)
    df["Predicted Volume"] = y_pred
    df["Lower Limit"] = y_pis[:, 0]
    df["Upper Limit"] = y_pis[:, 1]

    confidence = (1 - alpha) * 100 

    display_cols = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "month", "weekday", "hour",
                    "Predicted Volume", "Lower Limit", "Upper Limit"]

    st.subheader(f"Prediction Results with {confidence:.0f}% Prediction Interval")
    st.dataframe(df[display_cols])

st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted Vs. Actual", "Coverage Plot"])

# Tab 1: Feature Importance 
with tab1:
    st.write("Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")

# Tab 2: Histogram of Residuals
with tab2:
    st.write("Histogram of Residuals")
    st.image('all_residuals.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")

# Tab 3: Predicted Vs. Actual
with tab3:
    st.write("Predicted Vs. Actual")
    st.image('predicted_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")

# Tab 4: Coverage Plot
with tab4:
    st.write("Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confidence intervals.")