# Libraries
import pandas as pd
import numpy as np
import os
import time
import streamlit as st
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import math
from minisom import MiniSom # Self Organizing Maps
from tslearn.barycenters import dtw_barycenter_averaging # DWT: Dynamic Time Wrapping (Untuk Meninjau Kemiripan Antar Data Time Series)
from tslearn.clustering import TimeSeriesKMeans # KMeans Time Series
from sklearn.cluster import KMeans # KMeans Data General
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go  
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression # Linear Regression

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
# from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics


# Get list of files path .xlsx
def get_list_path():   
    list_files = [item for item in os.listdir('./dataset') if '.xlsx' in item and '$' not in item] # get list of filename
    list_path  = [r'./dataset/'+item for item in list_files] # create full path
    list_path = sorted(list_path)
    # print(list_path)   
    return list_path


# Get dataframe from files .xlsx
def get_df_list(input_file_paths):
    df_list = [] # initiate empty list
    for item in input_file_paths:
        dta = pd.read_excel(item)
        dta = dta.fillna(0.0) # fill NaN values with 0.0
        df_list.append(dta) # get scanned dataframe into list
    # print(df_list[1])
    return df_list


# Get list name of perusahaan
def get_list_perusahaan(input_df_list):
    list_p = [] # initiate empty list
    for df in input_df_list:
        for item in df.Payer.unique(): # get unique name of perusahaan each dataframe in column 'Payer'
            if item not in list_p:
                list_p.append(item)
    return list_p


# Get table sum
def get_existing_data_ts(input_df_list, list_p):
    table_sum = []
    # Initiate progress bar
    count = 0
    my_bar = st.progress(count, "Sedang Proses. Mohon Tunggu!")
    for payer in list_p:
        # Progress bar
        count = count+1
        time.sleep(0.01)
        my_bar.progress(int(100 * count / len(list_p)), f"Sedang Proses. Mohon Tunggu! Progress - {int(100 * count / len(list_p))}% import - {payer}")
        df_ = pd.DataFrame({'tahun': [2018, 2019, 2020, 2021, 2022], 'Emergency': 5 * [0],
                            'Health Checkup': 5 * [0], 'Inpatient': 5 * [0],
                            'Outpatient': 5 * [0]})
        df_.tahun = pd.to_datetime(df_.tahun, format='%Y').dt.year
        col_name = ["Emergency", "Health Checkup", "Inpatient", "Outpatient"]
        for i in range(len(input_df_list)):
            for pay in col_name:
                if payer in input_df_list[i]["Payer"].unique():
                    df_[pay][i] = input_df_list[i][pay][list(np.where(input_df_list[i]["Payer"] == payer)[0])].sum()
        df_["Total"] = df_["Emergency"] + df_["Health Checkup"] + df_["Inpatient"] + df_["Outpatient"]
        df_.set_index('tahun')
        table_sum.append([df_, payer])
    return table_sum


# Get Big Dataframe of Table Sum
def table_sum_big_data(table_sum_input, category):
    table_sum = deepcopy(table_sum_input)
    namesofMySeries = [d[1] for d in table_sum]

    # Initiate progress bar
    count = 0
    my_bar = st.progress(count, "Pengambilan Data Kategori. Mohon Tunggu!")
    for i in range(len(table_sum)):
        # Progress bar
        count = count+1
        time.sleep(0.01)
        my_bar.progress(int(100 * count / len(namesofMySeries)), f"Pengambilan Data Kategori - {category}. Mohon Tunggu! Progress - {int(100 * count / len(namesofMySeries))}% import - {namesofMySeries[i]}")
        table_sum[i][0] = table_sum[i][0] = table_sum[i][0].loc[:,[category]] # Choose category
        table_sum[i][0] = np.array(table_sum[i][0])
        table_sum[i][0]= table_sum[i][0].reshape(len(table_sum[i][0])) # Change the shape or dimension

    mySeries = [d[0] for d in table_sum] # Data Timeseries
    mySeries_col = [d[1] for d in table_sum] # Name Perusahaan
    dtf = pd.DataFrame(mySeries[0].T) # Transpose
    for i in range(1,len(mySeries)):
        dtf = pd.concat([dtf, pd.DataFrame(mySeries[i].T)], axis=1) # Join
    dtf.columns = mySeries_col
    dtf.index = ["2018", "2019", "2020", "2021", "2022"]
    return namesofMySeries, mySeries, dtf


# Get Big Dataframe of Table Sum 2
def table_sum_big_data2(table_sum_input, category):
    table_sum = deepcopy(table_sum_input)
    namesofMySeries = [d[1] for d in table_sum]

    # Initiate progress bar
    count = 0
    my_bar = st.progress(count, "Pengambilan Data Kategori dengan Normalisasi. Mohon Tunggu!")
    for i in range(len(table_sum)):
        # Progress bar
        count = count+1
        time.sleep(0.01)
        my_bar.progress(int(100 * count / len(namesofMySeries)), f"Pengambilan Data Kategori dengan Normalisasi - {category}. Mohon Tunggu! Progress - {int(100 * count / len(namesofMySeries))}% import - {namesofMySeries[i]}")
        table_sum[i][0] = table_sum[i][0] = table_sum[i][0].loc[:,[category]] # Choose category
        scaler = MinMaxScaler() # Normalisasi
        table_sum[i][0] = MinMaxScaler().fit_transform(table_sum[i][0])
        table_sum[i][0] = np.array(table_sum[i][0])
        table_sum[i][0]= table_sum[i][0].reshape(len(table_sum[i][0])) # Change the shape or dimension
    
    mySeries = [d[0] for d in table_sum] # Data Timeseries
    mySeries_col = [d[1] for d in table_sum] # Name Perusahaan
    dtf = pd.DataFrame(mySeries[0].T) # Transpose
    for i in range(1,len(mySeries)):
        dtf = pd.concat([dtf, pd.DataFrame(mySeries[i].T)], axis=1) # Join
    dtf.columns = mySeries_col
    dtf.index = ["2018", "2019", "2020", "2021", "2022"]
    return namesofMySeries, mySeries, dtf


def training_kmeans_sklearn(cluster_count, mySeries):
    pca = PCA(n_components=2) # Feature Reduction with PCA
    mySeries_transformed = pca.fit_transform(mySeries) # Feature Reduction 
    kmeans = KMeans(n_clusters=cluster_count,max_iter=5000)
    labels = kmeans.fit_predict(mySeries_transformed) # Labeling
    # st.write(labels)
    score = davies_bouldin_score(mySeries_transformed, labels)
    # st.write(mySeries_transformed)
    for i in range(len(kmeans.cluster_centers_)):
        st.write(f"cluster {i}: {kmeans.cluster_centers_[i]}")
    centroid = kmeans.cluster_centers_
    st.write(f"Nilai DBI: {score}")
    # st.write(kmeans.cluster_centers_)
    # Davies-Bouldin Index, closer to 0 better
    davies_bouldin_scores = []
    for k in range(2, cluster_count+1):
        km = KMeans(n_clusters=k,max_iter=5000)
        labels = km.fit_predict(mySeries_transformed) # Labeling
        score = davies_bouldin_score(mySeries_transformed, labels)
        davies_bouldin_scores.append(score)
    
    df_score = pd.DataFrame(
        {"Cluster": range(2, cluster_count+1), 
         "Davies Score": davies_bouldin_scores
         }
                        )
    # Create scatter plot
    fig = px.line(df_score,
                     x = "Cluster",
                     y = "Davies Score",
                     title = "Davies Bouldin Score")
                
    # Plot
    st.plotly_chart(fig, use_container_width=True)

    return mySeries_transformed, labels, centroid, score


def plot_cluster_kmeans_sklearn(centroid, 
                                df_ins,
                                namesofMySeries, 
                                cluster_count, 
                                mySeries, 
                                mySeries_transformed, 
                                labels):
    # st.write(labels)
    fig = go.Figure()

    df_f0 = pd.DataFrame({"x": mySeries_transformed[:, 0],
                         "y": mySeries_transformed[:, 1],
                         "cluster": [str(x) for x in labels],
                         "size": [6] * len(labels)}
                         )
    
    df_centroid = pd.DataFrame({
        "x": [x[0] for x in centroid],
        "y": [x[1] for x in centroid],
        "cluster": range(cluster_count),
        "size": [20] * cluster_count,
    })

    df_f = pd.concat([df_f0, df_centroid], axis=0)


    # Create scatter plot
    plt.figure(figsize=(25,10))
    fig = px.scatter(df_f,
                     x = "x",
                     y = "y",
                     color = "cluster",
                     size = "size",
                    #  color_discrete_sequence=["red", "blue", "green", "black"],
                     title = "Cluster Plot").update_layout(
                        template="plotly_dark",
                        plot_bgcolor='white',
                        )
    
    st.plotly_chart(fig)

    plot_count = math.ceil(math.sqrt(cluster_count))
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))

    fig, axs = plt.subplots(3,3,figsize=(25,25))
    fig.suptitle('Clusters')
    row_i=0
    column_j=0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
                if(labels[i]==label):
                    axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                    cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
    st.pyplot(plt.gcf())

    cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
    cluster_n = ["Cluster "+str(i) for i in range(cluster_count)]

    fancy_names_for_labels = [f"Cluster {label}" for label in labels]
    df_n = pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")

    # Define Average of Income 2018-2022
    df_n['Average'] = 0
    df_n['Min'] = 0
    df_n['Max'] = 0

    for item in namesofMySeries:
        df_n['Average'][item] = df_ins[item].mean()
        df_n['Min'][item] = df_ins[item].min()
        df_n['Max'][item] = df_ins[item].max()

    df_c = pd.DataFrame(
        {"x": cluster_n, 
         "y": cluster_c}
                        )
    # Create scatter plot
    fig = px.bar(df_c,
                     x = "x",
                     y = "y",
                     title = "Cluster Plot")
                
    # Plot
    st.plotly_chart(fig, use_container_width=True)

    # Show Dataframe
    st.dataframe(df_n)

    df_s = df_n.copy(deep=True)
    df_s.reset_index(inplace=True)
    df_s = df_s.rename(columns = {'index':'Perusahaan Asuransi'})
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(df_s)

    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
)


    fig = px.box(df_n, x="Cluster", y="Average", points=False)
    st.plotly_chart(fig, use_container_width=True)

    return df_n

# Generate dataframe result of clustering kmeans
def df_clust(
    labels,
    namesofMySeries,
    df_ins
    ):

    fancy_names_for_labels = [f"Cluster {label}" for label in labels]
    df_n = pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")

    # Define Average of Income 2018-2022
    df_n['Average'] = 0
    df_n['Min'] = 0
    df_n['Max'] = 0

    for item in namesofMySeries:
        df_n['Average'][item] = df_ins[item].mean()
        df_n['Min'][item] = df_ins[item].min()
        df_n['Max'][item] = df_ins[item].max()

    return df_n


# Timeseries Modelling

# Get table of big data
def table_sum_ts(cluster_count, df_n, table_sum):
    df_c_list = []
    for c in range(cluster_count):
        cluster_list = []
        for y in df_n.index:
            if df_n['Cluster'][y] == "Cluster " + str(c):
                cluster_list.append(y) 
        df_cluster = {}
        for i in cluster_list:
            for j in table_sum:
                if j[1] == i:
                    df_cluster[i] = j[0]
        df_c_list.append(df_cluster)
    return df_c_list

# Dataset New with Lag 1
def new_df_clust(df_clust_, num_lag, category):
    df_new_cluster_ = {}
    for i in df_clust_.keys():
        df_new_cluster_[i] = {}
        df_new_cluster_[i][f"Revenue_{category}"] = df_clust_[i][category]
        df_new_cluster_[i][f"Lag_{num_lag}"] = df_clust_[i][category].shift(num_lag)
        
    return df_new_cluster_


# Plot Training & Testing
def plot_ts_traintest(df_, num_lag, category):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=df_[f"Revenue_{category}"].index,
    y=df_[f"Revenue_{category}"],
    name = f"Revenue_{category}", # Style name/legend entry with html tags
    ))

    fig.add_trace(go.Scatter(
    x=df_[f"Lag_{num_lag}"].index,
    y=df_[f"Lag_{num_lag}"],
    name=f"Lag_{num_lag}",
    ))

    st.plotly_chart(fig)

# Training Process Linear Regression
def linear_regression(show_or_no, df, num_lag, category, year):
    df = pd.DataFrame(df)
    
    X = df[f"Lag_{num_lag}"].fillna(method = "bfill")  # features
    
    y = df[f"Revenue_{category}"]  # target
    y, X = y.align(X, join='inner')  # drop corresponding values in target
    X = X.values.reshape(-1,1)
    y = y.values

    # a LinearRegression instance and fit it to X and y.
    model = LinearRegression().fit(X,y)

    # Store the fitted values as a time series with the same time index as the training data
    y_pred = model.predict(X)

    df_ = pd.DataFrame({
        "Revenue True": y,
        "Revenue Predictions": y_pred,
    })

    if show_or_no:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
        x=year,
        y=df_["Revenue True"],
        name = "Revenue True", # Style name/legend entry with html tags
        ))

        fig.add_trace(go.Scatter(
        x=year,
        y=df_["Revenue Predictions"],
        name="Revenue Predictions",
        ))

        st.plotly_chart(fig)

    mae = metrics.mean_absolute_error(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse) # or mse**(0.5)

    if show_or_no:
        st.write(f"Error MAE: {mae}")
        st.write(f"Error RMSE: {rmse}")

    return y_pred




