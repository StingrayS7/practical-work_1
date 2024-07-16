import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.graph_objects as go

data = pd.read_csv('parkinsons.data')

all_features = data.loc[:,data.columns!='status'].values[:,1:]
out_come = data.loc[:,'status'].values

scaler = MinMaxScaler((-1,1))
X = scaler.fit_transform(all_features)
y = out_come

i = data.status.value_counts()

fig = go.Figure(data=[go.Bar(
            x=['Parkinson','Healthy'], y=i,
            text=i,
            textposition='auto',
        )])
fig.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)

xgb_pred = xgb_clf.predict(X_test)

print('Точность XGBoost на тренировочных данных : {:.2f}'.format(xgb_clf.score(X_train, y_train)*100))
print('Точность XGBoost на тестовых данных : {:.2f}'.format(xgb_clf.score(X_test, y_test)*100))

# Визуализация матрицы ошибок
cm = confusion_matrix(y_test, xgb_pred, labels=xgb_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=xgb_clf.classes_)
disp.plot()
plt.show()