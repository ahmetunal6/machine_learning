from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
train_data1=open("ekonomi.txt","r",encoding="utf8")
train1=train_data1.read()

train_data2=open("spor.txt","r",encoding="utf8")
train2=train_data2.read()

train_data3=open("teknoloji.txt","r",encoding="utf8")
train3=train_data3.read()

texts = [train1,train2,train3]


classes =['ekonomi', 'spor', 'teknoloji']

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(texts)


model = LinearSVC()
model.fit(vectors, classes)
train_data1=open("test.txt","r",encoding="utf8")
test=train_data1.read()

test_text = test
test_vector = vectorizer.transform([test_text])
prediction = model.predict(test_vector)
print("svm sınıflandırı ile test edilen veri{}ile ilgilidir".format(prediction)) 