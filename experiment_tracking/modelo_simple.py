# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


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
    print(f'Numero despues de oversampling: {Counter(y_over)}')

    return X_over, y_over

X, y = dataset_oversampling(X, y)

# Hacemos un split de nuestros datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos un modelo de Random Forest, y lo entrenamos
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Accuracy of the model: {metrics.accuracy_score(y_test, y_pred)}, '
      f'precision of the model: {metrics.precision_score(y_test, y_pred)}, '
      f'recall: {metrics.recall_score(y_test, y_pred)}')