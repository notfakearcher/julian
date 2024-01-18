# %% [markdown]
# # Application: Cluster Master

# %% [markdown]
# Application Description: 
#   - Takes in a dataset
#   - Allows users to select specific target columns
#   - Generate clusters based on selected columns
#   - Make predicions to cluster labels based on newly uploaded data
# 
# Inputs:
#   1. File (.csv) with 1st row as header [N * M]
# 
# Outputs:
#   1. Cluster Results Table [N * (M + 1)]
# 
# ---
# ---
# Version.Release: 
#   - 01.02
# 
# Author: 
#   - Julian Archer
# 
# Links: 
#   - Github: https://github.com/notfakearcher/julian
#   - LinkedIn: https://www.linkedin.com/in/archerj

# %% [markdown]
# ## Import Python Libraries

# %%
import sys
import streamlit as st
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans, OPTICS, DBSCAN, HDBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.utils import resample

# %% [markdown]
# ## Global Variables

# %%
# random state 
random_state = 4781

# colors
c_scale = 256
c1 = (1, 1, 1) # grey
c2 = (1, 0, 0) # red
c3 = (181/c_scale, 204/c_scale, 6/c_scale) # yellow
c4 = (6/c_scale, 122/c_scale, 204/c_scale) # blue
c5 = 'black' # black
c6 = ["#fbfce6", "#a2daf2", "#ffc7c7", '#8e6a9e']
c7 = [(0, 0, 1), (0, 1 ,0), (1, 0, 0), (1, 1, 0)]
c8 = ['#fcba03', '#0367fc', '#9003fc', '#fc3503', '#524e4d', '#35dbc0']

# %% [markdown]
# ## Application Header

# %%
# make streamlit application go into "wide" mode
# st. set_page_config(layout="wide")

# application title
st.title('Cluster Master - Release 01.02')

# application description
txt = 'This app takes in a dataset, \
  allows users to select specific target columns, \
  generates clusters based on selected columns, \
  and makes predicions to cluster labels based on newly uploaded data.'
st.markdown(txt)

# %% [markdown]
# ## Import Dataset

# %%
# upload file to generate clusters from
txt = 'Select and upload .csv file for generating clusters'
filepath1 = st.file_uploader(label = txt, accept_multiple_files = False, 
    label_visibility = "collapsed"
)

# if no file uploaded then exit operations
if filepath1 is not None:
    # successful file upload message
    st.markdown('Let the analysis begin !!!')
    
    # Hide filename on user interface
    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)
    df0 = pd.read_csv(filepath1, header = 0)
else:
    sys.exit()
     
# select which x columns to include
txt = 'Select which columns to include in analysis'
options = df0.columns.values.copy()

# identify X_cols
# X_cols = df0.columns
X_cols = st.multiselect(label = txt, options = options)

# if no options for columns selected then exit operations
if len(X_cols) < 1:
  # stop application from further execution
  st.stop()
else:
  # update intial dataframe
  df0 = df0[X_cols]

# standardize column names
df1 = df0.copy()
zeros = len(str(len(X_cols)))
temp = (np.arange(0, len(X_cols)) + 1).astype('str')
X_col_map = {'X' + temp[i].zfill(zeros):df0.columns[i] for i in range(len(temp))}
# X_cols = ['X' + i.zfill(zeros) for i in temp]
X_cols = X_col_map.keys()
df1.columns = X_cols

# encode categorical values to numerical values
le = LabelEncoder()

categorical_cols = df1[X_cols].select_dtypes("object").columns
df1[categorical_cols] = df1[categorical_cols].apply(le.fit_transform)

# %% [markdown]
# ## Data Cleaning and Transformation

# %%
# predictor columns
X_cols = df1.columns

# remove rows with missing valus
for X in df1.columns:
  cond1 = ~((df1[X].isna()))
  df1 = df1.loc[cond1,:]

# define X (feature matrix)
X = df1

 # normalize all X_cols
df2 = df1.copy()
norm = StandardScaler()
df2[X_cols] = norm.fit_transform(df2[X_cols])

# # remove outliers from each column
# for X in X_cols:
#   cond1 = ~((df2[X] >= 2) | (df2[X] <= -2))
#   df2 = df2.loc[cond1,:]

# %% [markdown]
# ## Visualize Dataset

# %%
# figure setup
fig_cols = 3
fig_rows = np.ceil(len(X_cols)/fig_cols).astype('int')
y_max = np.round(df2[X_cols].max().max(), 0)
y_min = np.round(df2[X_cols].min().min(), 0)
palette2 = [c3, c4]
fig_height = 5 * fig_rows
figsize = (15, fig_height )

# subplot of y vs each X (stripplot + violinplot + boxenplot)
f1, axes = plt.subplots(fig_rows, fig_cols, figsize = figsize)

# reshape axes to always use [row, col] index
axes = axes.reshape((fig_rows,fig_cols))

itr = 0
for x in X_cols:
  row = np.floor(itr/fig_cols) 
  row = row.astype('int')
  col = np.mod(itr, fig_cols)
  col = col.astype('int') 
  p1 = sns.histplot(ax = axes[row, col], data = df2, x = x, color = c2, linewidth = 1)
  axes[row, col].legend_ = None
  itr = itr + 1

# add supblot title 
f1.suptitle("Normalized Distribution Plot: Individual Features",
            y = 0.999, 
            fontsize = 20
)
f1.tight_layout()

# show column map reference
st.write(X_col_map)

# generate plot in application
with st.spinner(text = 'Trying to generate graph...Please wait a moment!!!'):
  st.pyplot(f1)

# %% [markdown]
# ## Train ML Model

# %%
# define X - feature matrix
X = df2[X_cols]

# split data into train and test datasets
X_train, X_test = train_test_split(X,
  train_size = 0.7, random_state = random_state
)

# only execute principal component analysis if 2 columns or more

if len(X_cols) >= 2:
  # get principal components
  pca = PCA(n_components = 2)
else:
  st.stop()

# fit principal components to training data
temp = pca.fit(X_train)

# pca explained variance ration
var_ratio = pca.explained_variance_ratio_

# transform data based on principal components
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
X = pca.transform(X)

# function to plot dendogram to select optimal number of clusters
def jra_plot_dendrogram(X, title, metric, method):
  # perform hierarchial clustering
  # Notes given [i, j]:
  # j = 1: Z[i, 0] cluster
  # j = 2: Z[i, 1] cluster
  # j = 2: distance between clusters Z[i, 0] and Z[i, 1]
  # j = 3: number of original observations in the newly formed cluster
  Z = linkage(X, method = method, metric = metric)
  cutoff = Z[:, 2].max() * 0.5
  p1 = dendrogram(Z, 
    orientation = 'top', 
    show_leaf_counts = True,
    color_threshold = cutoff, 
    above_threshold_color = c5, 
    #  distance_sort = 'ascending'
    distance_sort = 'descending'
    #  distance_sort = False
  )
  # clean up plot
  plt.title(title)
  plt.xlabel('X - Values')
  plt.ylabel('Distance / Disimilarity')
  plt.axhline(y = cutoff, color = c2, linestyle = '--', linewidth = 1.2)
  set_link_color_palette(c8)
  # plt.show()
  
  return(p1)
  
# function for fitting clustering_hierarchical
def jra_clustering_hierarchical(n_clusters):
  
  # initialize hierarchical agglomerative clustering
  clustering_model = AgglomerativeClustering(
    n_clusters = n_clusters,
    metric = 'euclidean',
    linkage = 'ward'
  )
  
  # return clustering model
  return(clustering_model)


# function for fitting clustering_k_means
def jra_clustering_k_means(X):
  # get optimal number of clusters k
  silhouette_avg = []
  silhouette_std = []
  k_values = np.arange(2, 5)

  for k in k_values:
    # initialize clustering model
    clustering_model = clustering_model = KMeans(
      n_clusters = k, 
      n_init = 'auto', 
      random_state = random_state, 
      init = 'k-means++'
    )
    # fit model and predict labels
    y_hat = clustering_model.fit_predict(X)
    # silhoutte score for each sample
    silhouette_values = silhouette_samples(X, labels = y_hat)
    # silhoutte avg
    temp = silhouette_values.mean()
    silhouette_avg.append(temp)
    # silhoutte std
    temp = silhouette_values.std()
    silhouette_std.append(temp)
    
  # ratio of avg to std
  silhouette_avg_to_std = np.divide(silhouette_avg, silhouette_std)

  # get optimal k
  i = np.where(silhouette_avg_to_std == silhouette_avg_to_std.max())
  k = k_values[i][0]

  # initialize k-means clustering
  clustering_model = KMeans(
    n_clusters = k, 
    n_init = 'auto', 
    random_state = random_state, 
    init = 'k-means++'
  )
  # return clustering model
  return(clustering_model)


# function for fitting clustering_k_means
def jra_clustering_dbscan(X):
  
  # determine optimal eps neighborhood value using k-distance
  nn = NearestNeighbors(n_neighbors = 2)
  temp = nn.fit(X)
  distances, indicies = nn.kneighbors(X)
  distances = np.sort(distances[:,1])
  change_points = zip(distances[:-1], distances[1:])
  distance_changes = [np.abs(b-a) for a, b, in change_points]
  max_change = np.max(distance_changes)
  cond = np.where(distance_changes == max_change)[0] - 1
  eps = distances[cond][0].astype('float')
  eps = eps if eps > 0.0 else 0.01
  
  # initialize k-means clustering
  clustering_model = DBSCAN(
    eps = eps,
    metric = 'euclidean'
  )
  
  # return clustering model
  return(clustering_model)


# select which clustering analysis method to apply on dataset
txt = 'Select clustering method to apply on dataset'
options = [
  'DBSCAN',
  'Hierarchical',
  'K-Means'
]
clustering_method = st.selectbox(label = txt, options = options, index = 1)


# get clustering model need to fit and generate labels
if clustering_method == 'Hierarchical':
  
  # plot dendrogrm for training and test to evaluate optimal (k) number of clusters
  f2, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 5))
  metric = 'euclidean'
  method = 'ward'

  # training data dendrogram
  title = 'Training Data\nDendrogram for Hierarchial Clustering - Divisive'
  plt.subplot(131)
  p1 = jra_plot_dendrogram(X_train, title, metric = metric, method = method)

  # testing data dendrogram
  title = 'Testing Data\nDendrogram for Hierarchial Clustering - Divisive'
  plt.subplot(132)
  p2 = jra_plot_dendrogram(X_test, title, metric = metric, method = method)

  # all data dendrogram
  title = 'All Data\nDendrogram for Hierarchial Clustering - Divisive'
  plt.subplot(133)
  p3 = jra_plot_dendrogram(X, title, metric = metric, method = method)

  # add supblot title 
  f2.suptitle("Dendrogram Plot: Based on Principal Components 1 and 2",
              y = 0.999, 
              fontsize = 20
  )
  f2.tight_layout()

  # show dendrogram plot
  with st.spinner(text = 'Trying to generate graph...Please wait a moment!!!'):
    st.pyplot(f2)
  
  # extract color list from dendrogram output object
  cluster_colors = p3['color_list']
  # determine optimal k based on threshold cutoff (inherent to color map)
  k = len(np.unique(cluster_colors)) - 1  
  # get clustering model
  clustering_model = jra_clustering_hierarchical(n_clusters = k)
  
elif clustering_method == 'K-Means':
  
  # get clustering model
  clustering_model = jra_clustering_k_means(X = X_train)

elif clustering_method == 'DBSCAN':
  
  # get clustering model
  clustering_model = jra_clustering_dbscan(X = X_train)
  
else:
  st.stop()



# function to plot clusters
def jra_plot_clusters(X, clustering_model, title):
  
  # fit clustering model
  temp = clustering_model.fit(X)
  
  # get predicted labels
  y_hat = clustering_model.labels_.tolist()

  # initialize plot paramaters
  X1_values = X[:, 0]
  X2_values = X[:, 1]
  X1_min = X1_values.min() - 1
  X2_min = X2_values.min() - 1
  X1_max = X1_values.max() + 1
  X2_max = X2_values.max() + 1


  # number of clusters to plot
  k = len(np.unique(y_hat))
    
  # try:
  #   k = clustering_model.n_clusters_
  # except:
  #   try:
  #     k = clustering_model.get_params()['n_clusters']
  #   except:
  #     try:
  #       st.stop()
  #     except: 
  #       st.stop()
  
  # plot clusters
  sns.scatterplot(x = X1_values, y = X2_values, hue = y_hat, palette = c8[0:k], s = 35)
  plt.title(title)
  plt.xlabel('PC1')
  plt.ylabel('PC2')
  
  # plt.show()
  return({'yhat': y_hat})
  
# plot clusters
f3, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 5))

# training data clusters
title = 'Training Data\nAgglomerative Clusters'
plt.subplot(131)
p1 = jra_plot_clusters(X_train, clustering_model, title)

# # testing data clusters
title = 'Testing Data\nAgglomerative Clusters'
plt.subplot(132)
p2 = jra_plot_clusters(X_test, clustering_model, title)

# # all data clusters
title = 'All Data\nAgglomerative Clusters'
plt.subplot(133)
p3 = jra_plot_clusters(X, clustering_model, title)

# add supblot title 
f3.suptitle("Cluster Plot: Based on Pricipal Components 1 and 2",
            y = 0.999, 
            fontsize = 20
)
f3.tight_layout()


# show cluster plot
with st.spinner(text = 'Trying to generate graph...Please wait a moment!!!'):
  st.pyplot(f3)


# %% [markdown]
# ## Output Final Results

# %% 
# get final cluster results table based on original data
get_i = df2.index.values
df_clusters = df1.copy()
df_clusters = df_clusters.loc[get_i]
df_clusters.columns = X_col_map.values()
df_clusters['Cluster'] = p3['yhat']

# show final cluster result table
st.write("Final Cluster Results")
with st.spinner(text = 'Trying to generate graph...Please wait a moment!!!'):
  st.dataframe(data = df_clusters)
  
# download final cluster results
st.download_button(
  label = 'Download Cluster Results',
  data = df_clusters.to_csv(index = False).encode('utf-8'),
  file_name = 'cluster_results.csv',
  mime = 'text/csv'
)