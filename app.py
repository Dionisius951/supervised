import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load model dan scaler
rf = joblib.load("./model/random_forest_regression.pkl")
lr = joblib.load("./model/liniearRegression.pkl")
gb = joblib.load("./model/gb_regressor.pkl")

# Using object notation
add_selectbox = st.sidebar.title(
    "Selamat Datang",
)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Pilih Opsi Algoritma Prediksi",
        ("Linear Regression", 'Gradient Boosting Regression', 'Random Forest Regression')
    )
    
def HandleModel(model) :
    match model:
        case 'Linear Regression':
            return lr
        case 'Random Forest Regression':
            return rf
        case 'Gradient Boosting Regression':
            return gb
            
model = HandleModel(add_radio)
# Title aplikasi
st.title(f"Prediksi Harga Mobil ({add_radio})")

# Input pengguna
CarName = st.text_input('Nama Mobil', placeholder="Inputkan nama mobil")
driveWheel = st.selectbox("Penggerak Mobil", ("RWD", "FWD", "4WD"), index=0)
engineLocation = st.selectbox("Posisi Mesin", ("Depan", "Belakang"), index=0)
wheelbase = st.number_input("Wheelbase (dalam cm)", step=0.1)
carwidth = st.number_input("Lebar Mobil (dalam cm)", step=0.1)
carlength = st.number_input("Panjang Mobil (dalam cm)", step=0.1)
curbweight = st.number_input("Curb Weight (dalam cm)", step=0.1)
engineSize = st.number_input("Ukuran Mesin (cc)", step=0.1)
borerasio = st.number_input("Bore Rasio (dalam unit)", step=0.1)
horsepower = st.number_input("Horse Power (HP)", step=0.1)

# Tombol prediksi
if st.button("Prediksi"):
    # Label encoder untuk kolom kategorikal
    label_encoder = LabelEncoder()

    # Data input
    input_data = pd.DataFrame([{
        "CarName": CarName,
        "drivewheel": driveWheel.lower(),
        "enginelocation": engineLocation.lower(),
        "wheelbase": wheelbase,
        "carlength": carlength,
        "carwidth": carwidth,
        "curbweight": curbweight,
        "enginesize": engineSize,
        "boreratio": borerasio,
        "horsepower": horsepower,
    }])

    # Encode kolom kategorikal
    input_data['CarName'] = label_encoder.fit_transform(input_data['CarName'])
    input_data['drivewheel'] = label_encoder.fit_transform(input_data['drivewheel'])
    input_data['enginelocation'] = label_encoder.fit_transform(input_data['enginelocation'])


    # Prediksi menggunakan model
    prediction = model.predict(input_data)

    # Tampilkan hasil
    st.success(f"Prediksi harga mobil menggunakan algoritma {add_radio} adalah : ${prediction[0]:,.2f}")


    