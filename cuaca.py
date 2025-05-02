import streamlit as st
import tensorflow as tf  
import numpy as np
import joblib

# Load model dan preprocessing
scaler = joblib.load('scaler.pkl')  
label_encoder = joblib.load('label_encoder.pkl')
interpreter = tf.lite.Interpreter(model_path="data_cuaca.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup halaman
st.set_page_config(page_title="Prediksi Cuaca", page_icon="🌤️", layout="centered")

# CSS Custom Background & Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #a2c2f4, #f0f9ff);
    }
    .stApp {
        background-image: linear-gradient(120deg, #a2c2f4 0%, #f0f9ff 100%);
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        padding: 2rem;
        border-radius: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-out;
    }
    .title {
        font-size: 48px;
        font-weight: 600;
        color: #2563eb;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        transition: color 0.3s ease;
    }
    .title:hover {
        color: #1d4ed8;
        transform: scale(1.05);
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #444;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .result {
        font-size: 26px;
        text-align: center;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        font-weight: 600;
        transition: transform 0.3s ease-in-out;
    }
    .result.sunny {
        background-color: #fff3cd;
        color: #856404;
    }
    .result.rainy {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .result.cloudy {
        background-color: #f8d7da;
        color: #721c24;
    }
    .result:hover {
        transform: scale(1.05);
    }
    .input-container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        gap: 20px;
    }
    .input-container div {
        flex: 1;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 16px;
        transition: background-color 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background-color: #1d4ed8;
        transform: scale(1.05);
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌤️ Prediksi Cuaca</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan parameter lingkungan untuk mengetahui kondisi cuaca.</div>', unsafe_allow_html=True)

# Input form
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.slider("🌡️ Suhu Udara (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
        air_pressure = st.slider("📈 Tekanan Udara (hPa)", min_value=800.0, max_value=1100.0, value=1013.0, step=0.1)

    with col2:
        relative_humidity = st.slider("💧 Kelembapan Relatif (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        avg_wind_speed = st.slider("💨 Kecepatan Angin Rata-rata (m/s)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

# Tombol prediksi
if st.button("🔍 Prediksi Cuaca"):
    try:
        input_data = np.array([[air_temp, relative_humidity, air_pressure, avg_wind_speed]])
        input_scaled = scaler.transform(input_data).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = np.argmax(prediction)
        weather_label = label_encoder.inverse_transform([predicted_index])[0]

        if weather_label == 'Sunny':
            st.markdown(f'<div class="result sunny">🌞 Prediksi kondisi cuaca: <strong>{weather_label.upper()}</strong></div>', unsafe_allow_html=True)
        elif weather_label == 'Rainy':
            st.markdown(f'<div class="result rainy">🌧️ Prediksi kondisi cuaca: <strong>{weather_label.upper()}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result cloudy">☁️ Prediksi kondisi cuaca: <strong>{weather_label.upper()}</strong></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
