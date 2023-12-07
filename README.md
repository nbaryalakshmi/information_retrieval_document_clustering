# information_retrieval_document_clustering
Document Clustering
---------------------

The input documents are collected from BBC news website. 104 documents are collected from categories such as technology, climate, sports, and health.

BBC news website link : https://www.bbc.co.uk/news

104 documents are collected from BBC news website from four categories such as technology, sports, health and climate. Data is vectorized using tf-idf vectorizer. K-means clustering is implemented to cluster the documents into four categories. Both the k-means model and tf.idf vectorizer is saved as pickle file. 

The frontend for clustering is developed using Dash-python. User enters an input query which needs to be clustered. This input is pre-processed using nltk libraries and passed to the tf-idf vectorizer and k-means clustering model, to predict the cluster of the input document. Both the vectorizer and the k-means model is retrieved from the pre-saved model pickle files. 

![image](https://github.com/nbaryalakshmi/information_retrieval_document_clustering/assets/127498506/67b48a88-6f0e-44f4-b5af-66c45130767e)
The above screenshot displays cluster output for user query “favourite sports: Football, Golf, Rugby” and the cluster seems correct.
