# train_model.py
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Cargar datos
car_evaluation = fetch_ucirepo(id=19)
df = car_evaluation.data.features.join(car_evaluation.data.targets)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Entrenar codificadores
encoders = {col: LabelEncoder().fit(df[col]) for col in df.columns}
df_encoded = df.copy()
for col in df.columns:
    df_encoded[col] = encoders[col].transform(df[col])

# Entrenar modelo
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']
model = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
model.fit(X, y)

# Guardar componentes
joblib.dump(model, 'car_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

print("Modelo y codificadores guardados!")