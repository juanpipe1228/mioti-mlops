# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import mlflow
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('hyperopt-exp')

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

space = {
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.choice("max_depth", [1, 2, 3, 5, 8]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
}

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag('model', 'Random Forest')
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', metrics.precision_score(y_test, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test, y_pred))

    return {'loss': 1 - metrics.recall_score(y_test, y_pred), 'status': STATUS_OK}

best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )