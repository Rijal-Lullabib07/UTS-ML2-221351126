import streamlit as st
import tensorflow as tf  
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')  
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="data_cuaca.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Cuaca")
st.write("Masukkan parameter lingkungan untuk memprediksi kondisi cuaca.")

# Form input pengguna
air_temp = st.number_input("Suhu Udara (°C)", min_value=-10.0, max_value=50.0, value=25.0)
relative_humidity = st.number_input("Kelembapan Relatif (%)", min_value=0.0, max_value=100.0, value=70.0)
air_pressure = st.number_input("Tekanan Udara (hPa)", min_value=800.0, max_value=1100.0, value=1013.0)
avg_wind_speed = st.number_input("Kecepatan Angin Rata-rata (m/s)", min_value=0.0, max_value=20.0, value=2.0)

if st.button("Prediksi Cuaca"):
    try:
        # Preprocessing input
        input_data = np.array([[air_temp, relative_humidity, air_pressure, avg_wind_speed]])
        input_scaled = scaler.transform(input_data).astype(np.float32)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Ambil hasil prediksi
        predicted_index = np.argmax(prediction)
        weather_label = label_encoder.inverse_transform([predicted_index])[0]

        st.success(f"Prediksi kondisi cuaca: **{weather_label.upper()}**")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
