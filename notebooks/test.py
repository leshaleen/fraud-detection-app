# -------- CLEAN OUTPUT --------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------- IMPORTS --------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import pickle

# -------- LOAD DATA --------
df = pd.read_csv('data/creditcard.csv')

# -------- PREPROCESS --------
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Save scaler
with open('app/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# -------- SPLIT --------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- SMOTE --------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------- MODEL --------
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train_res.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------- TRAIN --------
model.fit(X_train_res, y_train_res, epochs=5, batch_size=32)

# -------- EVALUATE --------
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# -------- SAVE --------
model.save('app/fraud_model.h5')

print("TRAINING DONE")


