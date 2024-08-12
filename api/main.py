'''
Caso true:
{
    "Sex": 1,
    "Red_Pixel": 43.2555,
    "Green_pixel": 30.8421,
    "Blue_pixel": 25.9025,
    "Hb": 6.3
}

caso false:
{
    "Sex": 0,
    "Red_Pixel": 45.0994,
    "Green_pixel": 27.9645,
    "Blue_pixel": 26.9361,
    "Hb": 16.2
}
'''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load('../experiment_tracking/models/clasificador_anemia.pkl')

app = FastAPI()

# Definir el esquema del mensaje esperado con BaseModel
class AnemiaRequest(BaseModel):
    Sex: int
    Red_Pixel: float
    Green_pixel: float
    Blue_pixel: float
    Hb: float

def data_prep(message: AnemiaRequest):
    data_dict = message.dict()  # Convertir el mensaje en un diccionario
    return pd.DataFrame([data_dict])  # Convertir el diccionario en un DataFrame

def anemia_prediction(message: AnemiaRequest):
    # Preparar los datos
    data = data_prep(message)
    label = model.predict(data)
    return {'Is Anemic?': int(label)}

@app.get('/')
def main():
    return {'message': 'Anemia Prediction'}

@app.post('/anemia-prediction/')
def predict_anemia(message: AnemiaRequest):
    model_pred = anemia_prediction(message)
    return model_pred
