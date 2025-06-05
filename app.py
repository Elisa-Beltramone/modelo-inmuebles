import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st
st.title("Modelo de precios de inmuebles")

# 1. Leer el dataset
df = pd.read_csv("Dataset_original.csv", dtype=str)

# 3. Filtrar solo operaciones de venta
df = df[df["Operacion"] == "Venta"]

# 4. Filtrar solo los inmuebles de interés
df = df[df["Inmueble"].isin(["Departamento", "Casa", "PH"])]

# 5. Separar columna Barrio_Ciudad en Barrio y Ciudad
df = df[df['Barrio_Ciudad'].str.contains('capital federal', case=False, na=False)]
barrio_ciudad = df["Barrio_Ciudad"].str.split(",", n=1, expand=True)
df["Barrio"] = barrio_ciudad[0]
df["Ciudad"] = barrio_ciudad[1].str.strip() if barrio_ciudad.shape[1] > 1 else ""


# 6. Limpiar nombres de barrio
df["Barrio"] = df["Barrio"].str.replace(r"Venta en |Alquiler en |Alquiler temporal en ", "", regex=True)

# 7. Quitar espacios extra en Ciudad
df["Ciudad"] = df["Ciudad"].str.strip()

# 8. Limpiar y convertir Latitud y Longitud a numérico
df["Latitud"] = pd.to_numeric(df["Latitud"].str.replace(",", "."), errors='coerce')
df["Longitud"] = pd.to_numeric(df["Longitud"].str.replace(",", "."), errors='coerce')

# 9. Filtrar solo Capital Federal y descartar alquiler temporal
df = df[(df["Ciudad"] == "Capital Federal") & (df["Operacion"] != "Alquiler Temporal")]

# 10. Limpiar columna Valor_Expensas, convertir a numérico y completar valores faltantes con 0
df["Valor_Expensas"] = pd.to_numeric(df["Valor_Expensas"].str.replace(r"[^0-9]", "", regex=True), errors='coerce').fillna(0)

# 11. Normalizar columna Ambientes
df["Ambientes"] = np.where(df["Ambientes"].str.contains("Monoambiente", na=False), "1", df["Ambientes"])
df["Ambientes"] = df["Ambientes"].str.extract(r"(\d+)", expand=False)
df["Ambientes"] = pd.to_numeric(df["Ambientes"], errors='coerce')
df = df[df["Ambientes"].notna()]

# 12. Eliminar columna Ciudad (ya es todo Capital Federal)
df = df.drop(columns=["Ciudad"], errors='ignore')

# 13. Extraer número de Sup_cubierta, Sup_Descubierta y Antiguedad
def extract_number(s):
    if pd.isna(s):
        return 0
    match = re.search(r"\d+[.,]?\d*", str(s))
    if match:
        return float(match.group(0).replace(",", "."))
    return 0

df["Sup_cubierta"] = df["Sup_cubierta"].apply(extract_number)
#df["Sup_Total"] = df["Sup_Total"].apply(extract_number)
df["Antiguedad"] = df["Antiguedad"].str.extract(r"(\d+)", expand=False).astype(float).fillna(0)

# 14. Agregar columna moneda y convertir valores a USD
tasa_usd_ars = 1200

df["Moneda"] = df["Valor_Inmueble"].str.extract(r"^([A-Za-z]+)")
df["Valor_Num"] = pd.to_numeric(df["Valor_Inmueble"].str.replace(r"[^0-9]", "", regex=True), errors='coerce')
df["Valor_USD"] = np.where(
    df["Moneda"] == "USD", df["Valor_Num"],
    np.where(df["Moneda"] == "ARS", df["Valor_Num"] / tasa_usd_ars, np.nan)
)
df = df[df["Valor_USD"].notna()]

# Agregar valor por metro cuadrado
df["valor_m2"] = df["Valor_USD"] / df["Sup_cubierta"]

# 15. Eliminar columnas innecesarias
df = df.drop(columns=["Operacion", "Latitud", "Longitud", "Lastmod", "Moneda", "Valor_Inmueble", "Valor_Num"], errors='ignore')

# 16. Eliminar outliers en Valor_USD
Q1 = df["Valor_USD"].quantile(0.25)
Q3 = df["Valor_USD"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df = df[(df["Valor_USD"] >= limite_inferior) & (df["Valor_USD"] <= limite_superior)]

# 17. Eliminar outliers en Antiguedad
Q1 = df["Antiguedad"].quantile(0.25)
Q3 = df["Antiguedad"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df = df[(df["Antiguedad"] >= limite_inferior) & (df["Antiguedad"] <= limite_superior)]

# 18. Eliminar outliers en Valor_Expensas
Q1 = df["Valor_Expensas"].quantile(0.25)
Q3 = df["Valor_Expensas"].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df = df[(df["Valor_Expensas"] >= limite_inferior) & (df["Valor_Expensas"] <= limite_superior)]

df['Barrio'] = df['Barrio'].str.replace(
    r'Venta en |Alquiler en |Alquiler temporal en ', '', regex=True
)

# Eliminar columnas innecesarias
df = df.drop(columns=["URL", "Sup_Total", "Sup_Descubierta"], errors='ignore')
df = df.drop(columns=["Barrio_Ciudad"], errors='ignore')

df = df[df["Sup_cubierta"] > 0]
#df["valor_m2"] = df["Valor_USD"] / df["Sup_cubierta"]

# 19. Verificar estructura y valores faltantes
print(df.info())
print(df.describe())
print(df.isna().sum())

# One-hot encoding y eliminación de columnas originales
df = pd.get_dummies(df, columns=["Barrio", "Estado", "Inmueble"], drop_first=False)


X = df.drop(columns=['Valor_USD'])
y = df['Valor_USD']

# 3. Ajustar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Predicciones y métricas básicas
y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("R^2:", r2)
print("MSE:", mse)
print("Intercepto:", modelo.intercept_)
print("Coeficientes:")
for name, coef in zip(X.columns, modelo.coef_):
    print(f"{name}: {coef}")




st.header("Ingresá los datos del inmueble para predecir el valor en USD")

# Obtén las opciones únicas de tu dataset para los selectbox
barrios = sorted([col.replace("Barrio_", "") for col in df.columns if col.startswith("Barrio_")])
estados = sorted([col.replace("Estado_", "") for col in df.columns if col.startswith("Estado_")])
inmuebles = sorted([col.replace("Inmueble_", "") for col in df.columns if col.startswith("Inmueble_")])

# Formulario de entrada
superficie = st.number_input("Superficie", min_value=5.0, max_value=5000.0,value=60.0)
min_ambientes = max(1, int((superficie - 37) // 7) + 1)
ambientes = st.number_input("Ambientes", min_value=min_ambientes, max_value=10, value=min_ambientes)
#ambientes = st.number_input("Ambientes", min_value=1, max_value=10, value=2)
st.info(
    "💡 **Sugerencia:** Para superficies muy grandes, la cantidad de ambientes debería ser acorde para obtener una estimación más realista."
)
antiguedad = st.number_input("Antigüedad (años)", min_value=0, max_value=100, value=20)
valor_expensas = st.number_input("Expensas (ARS)", min_value=0.0, max_value=1200500.0, value=5000.0)
barrio = st.selectbox("Barrio", barrios)
estado = st.selectbox("Estado", estados)
inmueble = st.selectbox("Tipo de inmueble", inmuebles)

# Botón para predecir, con validación de valor mínimo
if st.button("Predecir valor en USD"):
    # Construye un DataFrame con los datos ingresados
    input_dict = {
        "Sup_cubierta": [superficie],
        "Ambientes": [ambientes],
        "Antiguedad": [antiguedad],
        "Valor_Expensas": [valor_expensas],
    }
    # Agrega las columnas dummy necesarias (todas en 0, excepto la seleccionada en 1)
    for b in barrios:
        input_dict[f"Barrio_{b}"] = [1 if b == barrio else 0]
    for e in estados:
        input_dict[f"Estado_{e}"] = [1 if e == estado else 0]
    for i in inmuebles:
        input_dict[f"Inmueble_{i}"] = [1 if i == inmueble else 0]
    # Crea el DataFrame de entrada
    input_df = pd.DataFrame(input_dict)
    # Asegúrate de que las columnas estén en el mismo orden que X
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    # Predice
    prediccion = modelo.predict(input_df)[0]
    # Validación de valor mínimo
    if prediccion < 20000:
        st.error("El valor estimado es inferior a 20,000 USD. No se muestran predicciones por debajo de ese valor.")
    else:
        st.success(f"El valor estimado en USD es: {prediccion:,.2f}")
