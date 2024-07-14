import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Doc du lieu tu thu muc
raw_data_path_adl = 'data/adl/'
raw_data_path_fall = 'data/fall/'
data_list = []

# Doc va gan du lieu ADL
for filename in os.listdir(raw_data_path_adl):
    if filename.endswith('.csv'):
        file_path = os.path.join(raw_data_path_adl, filename)
        df = pd.read_csv(file_path, delimiter=';')
        df['Label'] = 'NotFall'
        data_list.append(df)

# Doc va gan du lieu Fall
for filename in os.listdir(raw_data_path_fall):
    if filename.endswith('.csv'):
        file_path = os.path.join(raw_data_path_fall, filename)
        df = pd.read_csv(file_path, delimiter=';')
        df['Label'] = 'Fall'
        data_list.append(df)

# Ket hop tat ca dataframe lai thanh mot dataframe
full_data = pd.concat(data_list, ignore_index=True)

# Loc cac cot can thiet
features = [' X-Axis', ' Y-Axis', ' Z-Axis']
X = full_data[features]
Y = full_data['Label']

# Chia du lieu thanh tap train va test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Chia tập train thành train và validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Luu tru du lieu
train_path = 'data/train/'
test_path = 'data/test/'
val_path = 'data/val/'

# os.makedirs(train_path, exist_ok=True)
# os.makedirs(test_path, exist_ok=True)
# os.makedirs(val_path, exist_ok=True)

X_train.to_csv(os.path.join(train_path, 'X_train.csv'), index=False)
Y_train.to_csv(os.path.join(train_path, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(test_path, 'X_test.csv'), index=False)
Y_test.to_csv(os.path.join(test_path, 'y_test.csv'), index=False)
X_val.to_csv(os.path.join(val_path, 'X_val.csv'), index=False)
Y_val.to_csv(os.path.join(val_path, 'y_val.csv'), index=False)

# Load dữ liệu đã xử lý
x_train = pd.read_csv(os.path.join(train_path, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(train_path, 'y_train.csv')).values.ravel()
x_test = pd.read_csv(os.path.join(test_path, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(test_path, 'y_test.csv')).values.ravel()
x_val = pd.read_csv(os.path.join(val_path, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(val_path, 'y_val.csv')).values.ravel()

# Traning mô hình
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = dt_model.predict(x_test)
print(classification_report(y_test, y_pred))

# Đánh giá trên tập validation
y_val_pred = dt_model.predict(X_val)
print(classification_report(y_val, y_val_pred))



# Du doan du lieu test
# y_pred = dt_model.predict(x_test)

#Performance evaluation
# def print_scores(y_true, y_pred):
#   print(classification_report(y_true, y_pred))
#
# # in ra ket qua
# print_scores(y_test, y_pred)