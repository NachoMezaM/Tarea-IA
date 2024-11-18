import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset
file_path = 'Iris.csv' 
iris_data = pd.read_csv(file_path)

# Preparar los datos
X = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Dividir conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo base
base_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Crear el modelo de Bagging
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)

# Entrenar el modelo
bagging_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = bagging_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
