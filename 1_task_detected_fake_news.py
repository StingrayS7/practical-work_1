import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset = pd.read_csv('fake_news.csv')

X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset.label, test_size=0.2, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)

# Точности предсказания
accuracy = accuracy_score(y_test,y_pred)
print(f' {round(accuracy * 100,2)} %')

# Отчет классификации
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(classification_report(y_test, y_pred))

# Визуализация матрицы ошибок
cm = confusion_matrix(y_test, y_pred, labels=pac.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=pac.classes_)
disp.plot()
plt.show()
