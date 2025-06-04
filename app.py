import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression

# (Puedes copiar aquí tu pipeline de limpieza y entrenamiento, o cargar un modelo ya entrenado)

st.title("Predicción de precio de inmuebles en Capital Federal")

st.write("Completa los datos para predecir el valor en USD:")

# Ejemplo de formulario sencillo
sup_cubierta = st.number_input("Superficie cubierta (m2)", min_value=10, max_value=1000, value=50)
sup_total = st.number_input("Superficie total (m2)", min_value=10, max_value=1000, value=60)
ambientes = st.number_input("Ambientes", min_value=1, max_value=10, value=2)
antiguedad = st.number_input("Antigüedad (años)", min_value=0, max_value=100, value=20)
valor_expensas = st.number_input("Expensas (ARS)", min_value=0, max_value=100000, value=5000)
barrio = st.selectbox("Barrio", ["Almagro", "Belgrano", "Palermo", "..."])  # Agrega todos los barrios que quieras
estado = st.selectbox("Estado", ["Bueno", "Muy Bueno", "Excelente", "Regular", "A Refaccionar"])
inmueble = st.selectbox("Tipo de inmueble", ["Departamento", "Casa", "PH"])

# Cuando el usuario hace click en el botón
if st.button("Predecir valor en USD"):
    # Aquí deberías crear un DataFrame con los datos ingresados y aplicar el mismo procesamiento/one-hot encoding que usaste para entrenar el modelo
    # Por simplicidad, aquí solo mostramos un mensaje
    st.success("Aquí iría la predicción del valor en USD (completa la lógica según tu modelo)")
