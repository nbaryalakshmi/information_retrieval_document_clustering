import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
nltk.download("stopwords")
nltk.download("punkt")
stopwords = stopwords.words('english')
def clean_text(text):
    port_stemmer = PorterStemmer()
    lst_tokens = word_tokenize(text)
    text_cleaned = ""
    for word in lst_tokens:
        if word not in stopwords:
            text_cleaned += port_stemmer.stem(word) + " "
    return text_cleaned
def vectorize_data(traindata):
    vectorizer=TfidfVectorizer()
    data_vectorized = vectorizer.fit_transform(traindata)
    #save vectorizer
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    return data_vectorized
def plotwordcloud(df_results):
    for i in df_results.labels.unique():
        pltdf=df_results[df_results.labels==i]
        text=" ".join(pltdf.text.tolist())
        print("Cluster: ",i)         
        word_cloud = WordCloud(width = 700, height = 700,background_color ='white', 
                           min_font_size = 14).generate(text) 
        plt.figure(figsize = (8, 8)) 
        plt.imshow(word_cloud) 
        plt.axis("off")  
        plt.show()
def get_performance(data_vectorized,predicted):
    dbi = metrics.davies_bouldin_score(data_vectorized.toarray(), predicted)
    ss = metrics.silhouette_score(data_vectorized.toarray(), predicted , metric='euclidean')
    print("DBI Score: ", dbi, "\nSilhoutte Score: ", ss)
if __name__ == '__main__':
    df_data=pd.read_excel("news_data.xlsx")
    df_data['filtered_news']=df_data['News'].apply(clean_text)
    df_data.head()
    data_train=df_data['filtered_news'].tolist()
    #Using TfidfVectorizer, create vector of words
    data_vectorized=vectorize_data(data_train)
    print("Vectorized data:\n")
    print(data_vectorized)
    #train and fit kmeans model
    k=4 #data taken from 4 category, climate, technology, sports and health
    obj_model_kmeans = KMeans(n_clusters=k, init='k-means++',max_iter=500)
    obj_model_kmeans.fit(data_vectorized)
    predicted=obj_model_kmeans.labels_
    print("Predicted labels:\n",predicted)
    #save kmeans model
    with open('clustering_model.pickle', 'wb') as f:
        pickle.dump(obj_model_kmeans, f)
    df_results=pd.DataFrame({"text":data_train,"labels":predicted})
    plotwordcloud(df_results) 
    get_performance(data_vectorized,predicted)
    