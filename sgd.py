

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

categories = ['spor', 'ekonomi', 'teknoloji']
## data dosyalarımı train etmek için açıp okuyoruz

train_data1=open("ekonomi.txt","r",encoding="utf8")
train1=train_data1.read()

train_data2=open("spor.txt","r",encoding="utf8")
train2=train_data2.read()

train_data3=open("teknoloji.txt","r",encoding="utf8")
train3=train_data3.read()

## train edilcek datalarla başlıkları bir listeye ekledim
train_data = [train1,train2,train3]
train_labels = ["ekonomi","spor","teknoloji"]

# text sınıflandırması için bir pipeline oluşturduk
text_classification_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
])

# train verilerini kullanarak modeli yarattık
text_classification_pipeline.fit(train_data, train_labels)



# test ettiğimiz kod bloğu
test_data=[]
train_data1=open("test.txt","r",encoding="utf8")
test=train_data1.read()
test_data.append(test)
predicted_labels = text_classification_pipeline.predict(test_data)
print("sgd classifier test etmem için verdiğiniz veri {}ile ilgilidir".format(predicted_labels))



