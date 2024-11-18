import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Cargar el dataset
file_path = 'lechuzasdataset.csv'
lechuzas_data = pd.read_csv(file_path)

# Preparar las características (X) y la variable objetivo (y)
X = lechuzas_data.drop(columns=['id', 'Potencia'])
y = lechuzas_data['Potencia']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo base con ajuste de parámetros
base_model = DecisionTreeRegressor(max_depth=10, random_state=42)

# Crear el modelo de Bagging con más estimadores
bagging_model = BaggingRegressor(estimator=base_model, n_estimators=100, random_state=42)

# Entrenar el modelo
bagging_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = bagging_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")
