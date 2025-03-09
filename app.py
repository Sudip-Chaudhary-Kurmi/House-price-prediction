import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Streamlit page settings
st.set_page_config(page_icon=":house:", page_title="House Price Prediction")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #1d2533; }
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
with st.sidebar:
    choose = option_menu("App Gallery", ["Prediction", "About", "DataSet"],
                         icons=['gear-wide', 'house', 'database-fill-add'],
                         menu_icon="app-indicator", default_index=0)

# About Section
def About():
    st.title("House Price Prediction App")
    st.write("""
    Welcome to the House Price Prediction App! This app predicts house prices based on different features.
    """)

    st.header("How to Use")
    st.write("""
    1. Enter the necessary house details in the Prediction tab.
    2. Click on the 'Predict' button to get an estimated price.
    3. The model will display the predicted price based on input features.
    """)

    st.header("Dataset Used")
    st.write("""
    The dataset includes details such as the number of bedrooms, house size, and age, which help in estimating the price.
    """)
    df = pd.read_csv('house_price_prediction_dataset.csv')
    st.title("Dataset")
    st.dataframe(df)

# Prediction Section
def prediction():
    df = pd.read_csv('house_price_prediction_dataset.csv')
    df = df.drop(columns=["proximity_to_city_center", "neighborhood_quality", "lot_size"])

    X = df.drop(columns="house_price")
    y = df["house_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)

    # Streamlit UI
    st.title('House Price Prediction')

    num_bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, max_value=10, value=3)
    num_bathrooms = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=5, value=2)
    square_footage = st.number_input("Enter the square footage:", min_value=300, max_value=10000, value=1500)
    age_of_house = st.number_input("Enter the age of the house:", min_value=0, max_value=150, value=20)

    user_input = pd.DataFrame({
        'num_bedrooms': [num_bedrooms],
        'num_bathrooms': [num_bathrooms],
        'square_footage': [square_footage],
        'age_of_house': [age_of_house]
    })

    if st.button('Predict House Price'):
        predicted_price = mlr.predict(user_input)
        st.header(f'Predicted House Price: ${predicted_price[0]:,.2f}')

# Dataset Section
def dataset():
    df = pd.read_csv('house_price_prediction_dataset.csv')
    st.title("Dataset")
    st.dataframe(df)

# Page Navigation Logic
if choose == "About":
    About()
elif choose == "Prediction":
    prediction()
elif choose == "DataSet":
    dataset()
