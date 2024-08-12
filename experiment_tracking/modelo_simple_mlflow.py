# Importing necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import warnings
import mlflow
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('simple-exp')

df = pd.read_csv('anemia_prediction_data.csv')

# Procesar datos
df = df.drop('Number', axis = 1) 
df.Sex = df.Sex.str.strip()

enc = LabelEncoder()
df['Sex'] = enc.fit_transform(df['Sex'])
df['Anaemic'] = enc.fit_transform(df['Anaemic'])

# X y 
X = df.drop(['Anaemic'], axis=1)
y = df['Anaemic']


def dataset_oversampling(X, y):
    # Definimos la estrategia de oversampling
    over = RandomOverSampler(sampling_strategy=1)
    # Adaptamos a nuestro dataset
    X_over, y_over = over.fit_resample(X, y)
    # summarize class distribution
    print(f'Numero de casos despues de oversampling: {Counter(y_over)}')

    return X_over, y_over

X, y = dataset_oversampling(X, y)

# Hacemos un split de nuestros datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos un modelo de Random Forest, y lo entrenamos
with mlflow.start_run():
    mlflow.set_tag('Author', 'Juan Felipe')
    mlflow.set_tag('Model', 'Random Forest')

    # Log param information
    # Logeamos sobre los datos
    n_estimators = 100
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=8,
                                 criterion='gini', random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mlflow.log_metric('accuracy', metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', metrics.precision_score(y_test, y_pred))
    mlflow.log_metric('recall', metrics.recall_score(y_test, y_pred))



    with open('models/clasificador_anemia.pkl', 'wb') as f_out:
        pickle.dump(clf, f_out)

    mlflow.log_artifact(local_path='models/clasificador_anemia.pkl', artifact_path='models_pickle')