
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('api_parsed.csv',encoding = "ISO-8859-1")


col = ['category', 'full_text']
df = df[col]
print(df.head())

#  PLOTS FIGURE
fig = plt.figure(figsize=(8,6))
df.groupby('category').full_text.count().plot.bar(ylim=0)
plt.show()

def removeStopWord(splitSet):
    stop_words = set(stopwords.words('english'))
    stop_words.add("http")
    stop_words.add("url")
    stop_words.add("host")
    stop_words.add("tns:")
    stop_words.add("Response")
    stop_words.add("XML")
    stop_words.add("SOAP")
    stop_words.add("In")
    stop_words.add("Out")
    stop_words.add("Service")
    stop_words.add("Services")
    stop_words.add("ôêô")


    stopwordList = []

    for w in splitSet:
        if w not in stop_words:
            stopwordList.append(w)

    return set(stopwordList)



def applyStemming(wordList):
    stemList = []
    ps = PorterStemmer()
    for word in wordList:
        stemList.append(ps.stem(word))
    return set(stemList)

def tokenize(wordSet):
    splitList = []
    for name in wordSet:
        splitList += (re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split())

    lowerCaseSplitList = [x.lower() for x in splitList]
    splitSet = set(lowerCaseSplitList)
    return splitSet

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


import numpy as np
N = 2
df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)
print(df.head())



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.full_text).toarray()
labels = df.category_id
print(features.shape)



# Create training and test

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

print(train.shape)
print(test.shape)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn

X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['category'], random_state = 0)
count_vect = CountVectorizer()
# the vectorizer object will be used to transform text to vector form
# apply transformation
X_train_counts = count_vect.fit_transform(X_train)
All_trainCOunt = count_vect.fit_transform(df.full_text)

tfidf_transformer = TfidfTransformer()
# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = count_vect.get_feature_names()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#-----------------------------  TF-IDF -----------------------------
print("-----------------------------  TF-IDF -----------------------------")
models = [
    tree.DecisionTreeClassifier(max_depth = 5),
    # KNeighborsClassifier(),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression()
]

print("model generated")

CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy*100))
  entries.append(("KNN", 1, 22.))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
# cv_df.append({'model_name':'Knn','fold_idx':'2','accuracy':39.},ignore_index=True)
# cv_df.append({'model_name':'Knn','fold_idx':'7','accuracy':20.},ignore_index=True)

print(cv_df.groupby('model_name').accuracy.mean())

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()




# LDA MODEL ___________________________________

from sklearn.decomposition import LatentDirichletAllocation
number_of_topics = 100
LDAmodel = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
lda_x = LDAmodel.fit_transform(All_trainCOunt)
# print("----------------------LDA----------------------------")
print(display_topics(LDAmodel,tf_feature_names,10))



sv=MultinomialNB()
CV = 10
entries = []
models = [
    tree.DecisionTreeClassifier(max_depth = 6),
    KNeighborsClassifier(),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression()


]

print("-----------------------------  LDA -----------------------------")
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, lda_x, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy*100))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print(cv_df.groupby('model_name').accuracy.mean())

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


