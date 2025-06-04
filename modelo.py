import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



# 1. Leer el dataset
df = pd.read_csv("Dataset_original.csv", dtype=str)

# 2. Eliminar columnas innecesarias
df = df.drop(columns=["URL", "Sup_Descubierta"], errors='ignore')


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
df["Sup_Total"] = df["Sup_Total"].apply(extract_number)
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


df = df.drop(columns=["Barrio_Ciudad"], errors='ignore')

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

