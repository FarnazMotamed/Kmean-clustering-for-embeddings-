import pandas as pd

import numpy as np
from time import time
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

dataframe_embeddings = pd.read_csv('embeddings_sv.csv', names=('id','content_id','language','publisher','date','embeddings'))
dataframe_embeddings['embeddings'] = dataframe_embeddings['embeddings'].apply(eval)

t0 = time()

x = np.array(dataframe_embeddings['embeddings'])
y =[np.array(x[i]).reshape(300,) for i in range(len(x))]
dataframe_embeddings['features'] = np.array(y)

X = np.array(y)

#to make the vectors normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

#apply kmeans with 20 clusters
kmeans = KMeans(n_clusters=20, random_state=0, max_iter=100).fit(X)
kmeans.fit_transform(X)

#kmeans.fit_predict(X)

print("done in %fs" % (time() - t0))
