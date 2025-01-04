import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score as sk_silhouette_score
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tslearn.clustering import silhouette_score as ts_silhouette_score
from sklearn.metrics import davies_bouldin_score as sk_davies_bouldin_score
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



# Calculate the BMI
def calculate_bmi(height, weight):
    return weight / (height / 100) ** 2

# Create folder if not exist
def createFolder(directory) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# Use Canvas to convert figure to image
def figure_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    return img

# Merge two plots and save the result
def coomparePlotAndSafe(plt_prev, plt_after, save_path) -> None:
    img_prev = figure_to_image(plt_prev)
    img_after = figure_to_image(plt_after)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(img_prev)
    axs[0].set_title('Before filling')
    axs[0].axis('off')
    axs[1].imshow(img_after)
    axs[1].set_title('After filling')
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# Used to check the consistency of a column in a DataFrame
global_data = None

def init_worker(data):
    global global_data
    global_data = data

def check_url_consistency(url_column_name):
    url, column_name = url_column_name
    # Accede alla variabile globale
    values = global_data.loc[global_data['_url'] == url, column_name].unique()
    if len(values) > 1:
        return f'Error: Different values for {column_name} in {url} {values}'
    return None

def checkDataConsistency(data, column_name, num_threads=None):
    unique_urls = data['_url'].unique()
    if num_threads is None:
        num_threads = os.cpu_count()
    # Create a list of tuples with the url and the column name
    args = [(url, column_name) for url in unique_urls]
    # Use ThreadPoolExecutor
    with ProcessPoolExecutor(max_workers=num_threads, initializer=init_worker, initargs=(data,)) as executor:
        results = list(tqdm(executor.map(check_url_consistency, args), 
                            total=len(unique_urls),
                            desc=f"Checking consistency for [{column_name}]"))
    return results

# Use the distribution of the column to fill the missing values
def fill_missing_values(data, column_name):
    # Calculate the distribution of the column
    distribution = data[column_name].value_counts(normalize=True)
    # Fill the missing values
    data[column_name] = data[column_name].apply(
        lambda x: np.random.choice(
            distribution.index, p=distribution.values) if pd.isnull(x) else x)

# Scatter plot of the data like x
# data is a list of numbers
# Reshape is used to add a dimension to the data
def scatter_plot(data, y, reshape=False):
    if reshape:
        data = np.array(data).reshape(-1, 1)
    # zeros_likes
    plt.plot(data, y, alpha=0.5)
    plt.show()

# Apply the BMI formula
def map_bmi(data):
    data['bmi'] = data.apply(lambda x: calculate_bmi(x['height'], x['weight']), axis=1)

# Take data and column name and print min and max values
def print_min_max(data, column_name):
    print(f"Min value for {column_name}: {data[column_name].min()}")
    print(f"Max value for {column_name}: {data[column_name].max()}")

# Take data and column name and return a list of unique values
def get_unique_values(data, column_name):
    return data[column_name].unique()

# Extract values from two columns
def extract_values(data, *args):
    features = []
    for arg in args:
        features.append(arg)
    #data = data.dropna()
    # Remove one list level
    features = [item for sublist in features for item in sublist]
    print(f'data: {data} features: {features}')
    dataset = data[features].values
    # Remove NaN values
    dataset = dataset[~np.isnan(dataset).any(axis=1)]
    return dataset

# Calculate SSE for each K in interver [1, max_k]
# Finally, plot the SSE values
# Take date
def plot_sse(data, max_k, reshape=False, title='SSE'):
    if reshape:
        data = np.array(data).reshape(-1, 1)
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title(title)
    plt.show()
    return sse

# Calculate the silhouette score for each K in interver [2, max_k]
# Finally, plot the silhouette score values
def plot_silhouette_score(data, max_k, reshape=False):
    if reshape:
        data = np.array(data).reshape(-1, 1)
    silhouette_scores = []
    #for k in range(2, max_k + 1):
    for k in tqdm(range(2, max_k + 1), desc='Calculating silhouette score'):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        cluster = kmeans.labels_
        silhouette_score = sk_silhouette_score(data, cluster)
        silhouette_scores.append(silhouette_score)
        print(f'Silhouette score for {k} clusters: {silhouette_score}')
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    return silhouette_scores

# Calculate the silhouette score for each K in interver [2, max_k] using tslearn
# This function is multithreaded
def plot_silhouette_score_multithread(data, max_k, reshape=False, n_jobs=1, number_of_samples=200000):
    if reshape:
        data = np.array(data).reshape(-1, 1)
    silhouette_scores = []
    #for k in range(2, max_k + 1):
    for k in tqdm(range(2, max_k + 1), desc='Calculating silhouette score'):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        cluster = kmeans.labels_
        silhouette_scores.append(ts_silhouette_score(data, cluster, sample_size=number_of_samples, n_jobs=n_jobs, metric='precomputed'))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# Calculate the davies bouldin score for each K in interver [2, max_k]
def plot_davies_bouldin_score(data, max_k, reshape=False):
    if reshape:
        data = np.array(data).reshape(-1, 1)
    davies_bouldin_scores = []
    for k in tqdm(range(2, max_k + 1), desc='Calculating Davies Bouldin score'):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        cluster = kmeans.labels_
        davies_bouldin_score = sk_davies_bouldin_score(data, cluster)
        davies_bouldin_scores.append(davies_bouldin_score)
        print(f'Davies Bouldin score for {k} clusters: {davies_bouldin_score}')
    plt.plot(range(2, max_k + 1), davies_bouldin_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies Bouldin Score')
    plt.show()
    return davies_bouldin_scores


# Normalize the data with StandardScaler
def normalize_data(data, scaler_type='standard'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    else:
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
# Convert the clusters to a DataFrame and assign each sample to a cluster
def convert_create_labels(data, clusters, hot_encode=False, columns=None):
    dataset_normalized_df = -1
    # Create a DataFrame with the normalized data
    if hot_encode:
        dataset_normalized_df = pd.DataFrame(data, columns=['bmi', 'position', 'cyclist_age', 'profile_1', 'profile_2', 'profile_3', 'profile_4', 'profile_5'])
    else:
        if columns is not None:
            dataset_normalized_df = pd.DataFrame(data, columns=columns)
        else:
            dataset_normalized_df = pd.DataFrame(data, columns=['bmi', 'position', 'cyclist_age', 'profile'])

    # Assign each sample to a cluster
    cluster_labels = [-1] * len(dataset_normalized_df)  # -1 means no cluster
    for cluster_idx, indices in enumerate(tqdm(clusters, desc="Assigning clusters")):
        for index in indices:
            cluster_labels[index] = cluster_idx
    # Add the cluster column to the dataset
    dataset_normalized_df['Cluster'] = cluster_labels

    return dataset_normalized_df

def calculate_best_k(silhouette_scores, davies_bouldins):
    normalized_silhouette = [(score - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) for score in silhouette_scores]
    normalized_davies = [(min(davies_bouldins) - score) / (min(davies_bouldins) - max(davies_bouldins)) for score in davies_bouldins]

    combined_scores = [silhouette + davies for silhouette, davies in zip(normalized_silhouette, normalized_davies)]
    best_k = combined_scores.index(max(combined_scores)) + 2

    return best_k

# This function is used to cluster the data using KMeans
def clustering_K_means(data, features, max_k, title='Clustering K_means', thd_plot=False, thd_view=False, normalize_type='standard', K=None):
    # Extract the values from the dataset
    X = extract_values(data, features)
    # Normalize the data
    X_normalized = normalize_data(X, scaler_type=normalize_type)
    # Plot the SSE
    plot_sse(X_normalized, max_k)
    # Plot the silhouette score
    silhouette_scores = plot_silhouette_score(X_normalized, max_k, False)
    # Plot the davies bouldin score
    davies_bouldins = plot_davies_bouldin_score(X_normalized, max_k, False)
    # From the two best calculated K
    best_k = calculate_best_k(silhouette_scores, davies_bouldins)
    if K is not None:
        n_clusters = K
    else:
        n_clusters = best_k
    print(f'Number of clusters: {n_clusters}')
    # Create the KMeans model
    KMeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    # Fit the model
    KMeans_model.fit(X_normalized)
    # Plot
    n_clusters = len(np.unique(KMeans_model.labels_))
    # features = ['', '']
    X_normalized_df = pd.DataFrame(X_normalized, columns=features)
    # Aggiungi la colonna con le etichette dei cluster
    X_normalized_df['Cluster'] = KMeans_model.labels_

    # Usa pairplot per visualizzare ogni coppia di feature
    sns.pairplot(X_normalized_df, hue='Cluster', palette=sns.color_palette(n_colors=n_clusters))
    # Add features to the title
    str_features = ' - '
    for feature in features:
        str_features += feature + ' '
    plt.suptitle(title+str_features, y=1.02)
    plt.show()

    if features.__len__() == 3:
        if thd_plot: # 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_normalized_df[features[0]], 
                           X_normalized_df[features[1]], 
                           X_normalized_df[features[2]], 
                           c=KMeans_model.labels_, 
                           cmap='viridis')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                ax.set_title("3D - " + title + str_features)
                plt.show()
        if thd_view: # 3D plot with plotly
            fig = px.scatter_3d( X_normalized_df,
                x=features[0],
                y=features[1],
                z=features[2],
                color='Cluster',
                title="3D - " + title + str_features,
                labels={
                    features[0]: features[0],
                    features[1]: features[1],
                    features[2]: features[2],
                    'Cluster': 'Cluster ID'
                }
            )

            # Custom layout
            fig.update_layout(
                legend=dict(
                    title='Cluster ID',  # Title of the legend
                    x=1,  # Position orizontal
                    y=1  # Position vertical
                )
            )

            # Show the plot 3D
            fig.show()
    else:
        if thd_plot or thd_view:
            print('Features must be 3!')
    

# This function is used to cluster the data using XMeans
def clustering_X_means(data, features, max_k, title='Clustering X_means', thd_plot=False, thd_view=False, seed=23, normalize_type='standard'):
    # Extract the values from the dataset
    X = extract_values(data, features)
    # Normalize the data
    X_normalized = normalize_data(X, scaler_type=normalize_type)
    # Create the X_means
    xmeans_instance = xmeans(X_normalized, kmax=max_k, random_state=seed)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    # Print clusters number
    print(f'Number of clusters: {len(clusters)}')
    #
    X_normalized_df = convert_create_labels(X_normalized, clusters, hot_encode=False, columns=features)
    # Plot
    plt.figure(figsize=(12, 8))
    sns.pairplot(
        X_normalized_df, 
        hue='Cluster', 
        palette='coolwarm', 
        corner=False, 
        plot_kws={'alpha':0.5, 's':20}
    )
    # Add features to the title
    str_features = ' - '
    for feature in features:
        str_features += feature + ' '
    plt.suptitle(title+str_features, y=1.02)
    plt.show()

    if features.__len__() == 3:
        if thd_plot: # 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_normalized_df[features[0]], 
                           X_normalized_df[features[1]], 
                           X_normalized_df[features[2]], 
                           c=X_normalized_df['Cluster'], 
                           cmap='viridis')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                ax.set_title("3D - " + title + str_features)
                plt.show()
        if thd_view: # 3D plot with plotly
            fig = px.scatter_3d( X_normalized_df,
                x=features[0],
                y=features[1],
                z=features[2],
                color='Cluster',
                title="3D - " + title + str_features,
                labels={
                    features[0]: features[0],
                    features[1]: features[1],
                    features[2]: features[2],
                    'Cluster': 'Cluster ID'
                }
            )

            # Custom layout
            fig.update_layout(
                legend=dict(
                    title='Cluster ID',  # Title of the legend
                    x=1,  # Position orizontal
                    y=1  # Position vertical
                )
            )

            # Show the plot 3D
            fig.show()
    else:
        if thd_plot or thd_view:
            print('Features must be 3!')


# This function is used to cluster the data using Agglomerate (Hierarchical Clustering)
def clustering_Agglomerate(data, features, max_k, title='Clustering Agglomerate', thd_plot=False, thd_view=False, 
                           normalize_type='standard', linkage_method='ward', metric='euclidean', K=None, plot_dendrogram=True):
    # Extract the values from the dataset
    X = extract_values(data, features)
    # Normalize the data
    X_normalized = normalize_data(X, scaler_type=normalize_type)
    # Plot the SSE
    plot_sse(X_normalized, max_k)
    # Plot the silhouette score
    silhouette_scores = plot_silhouette_score(X_normalized, max_k, False)
    # Plot the Davies-Bouldin score
    davies_bouldins = plot_davies_bouldin_score(X_normalized, max_k, False)
    # Determine the best number of clusters
    best_k = calculate_best_k(silhouette_scores, davies_bouldins)
    if K is not None:
        n_clusters = K
    else:
        n_clusters = best_k
    print(f'Number of clusters: {n_clusters}')
    
    # Create the dendrogram if required
    cluster_colors = {}
    if plot_dendrogram:
        print("Generating dendrogram...")
        Z = linkage(X_normalized, method=linkage_method, metric=metric)
        
        # Automatically calculate a threshold based on the desired number of clusters
        max_d = Z[-n_clusters, 2] if n_clusters is not None else 0.4

        plt.figure(figsize=(10, 7))
        dend = dendrogram(
            Z,
            truncate_mode='lastp',  # Show the last merges
            color_threshold=max_d,  # Threshold for cluster colors
            above_threshold_color='gray',  # Color above the threshold
        )
        plt.axhline(y=max_d, color='red', linestyle='--', label=f'Threshold: {max_d:.2f}')
        plt.title(f"Dendrogram ({linkage_method.capitalize()} Linkage)")
        plt.legend()
        plt.show()

        # Extract colors assigned by the dendrogram
        for leaf, color in zip(dend['leaves'], dend['leaves_color_list']):
            cluster_colors[leaf] = color

    # Create the AgglomerativeClustering model
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    # Create dataframe
    X_normalized_df = pd.DataFrame(X_normalized, columns=features)
    # Fit the model
    clusters = cluster.fit_predict(X_normalized_df)
    # Add the cluster labels to the dataframe
    X_normalized_df['Cluster'] = clusters

    # Create a colormap for the scatter plot
    unique_clusters = np.unique(clusters)
    colormap = get_cmap('tab10', len(unique_clusters))
    scatter_colors = [colormap(cluster_id) for cluster_id in unique_clusters]

    # Scatter plot with cluster colors matching the dendrogram
    # Add the cluster labels to the dataframe
    X_normalized_df['Cluster'] = clusters

    # Use pairplot to visualize each pair of features
    sns.pairplot(X_normalized_df, hue='Cluster', palette=sns.color_palette(n_colors=n_clusters))

    # Add features to the title
    str_features = ' - '
    for feature in features:
        str_features += feature + ' '

    plt.suptitle(title + str_features, y=1.02)
    plt.show()

    # 3D Visualization (if 3 features)
    if len(features) == 3:
        if thd_plot:  # 3D plot with matplotlib
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for cluster_id in unique_clusters:
                cluster_data = X_normalized_df[X_normalized_df['Cluster'] == cluster_id]
                ax.scatter(
                    cluster_data[features[0]], cluster_data[features[1]], cluster_data[features[2]],
                    color=scatter_colors[cluster_id], label=f"Cluster {cluster_id}"
                )
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(features[2])
            ax.set_title(f"3D - {title}")
            plt.legend()
            plt.show()
        if thd_view:  # 3D plot with Plotly
            import plotly.express as px
            fig = px.scatter_3d(
                X_normalized_df,
                x=features[0],
                y=features[1],
                z=features[2],
                color='Cluster',
                title=f"3D - {title}",
                labels={
                    features[0]: features[0],
                    features[1]: features[1],
                    features[2]: features[2],
                    'Cluster': 'Cluster ID'
                }
            )
            fig.update_layout(
                legend=dict(
                    title='Cluster ID',
                    x=1,
                    y=1
                )
            )
            fig.show()
    else:
        if thd_plot or thd_view:
            print('Features must be 3!')

# This function is used to cluster the data using DBSCAN
def clustering_DBSCAN(data, features, max_k, title='Clustering DBSCAN', thd_plot=False, thd_view=False, normalize_type='standard', eps=0.5, min_samples=5, leaf_size=30, n_jobs=None):
    # Extract the values from the dataset
    X = extract_values(data, features)
    # Normalize the data
    X_normalized = normalize_data(X, scaler_type=normalize_type)
    # Create the AgglomerativeClustering model
    cluster = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)
    # Create dataframe
    X_normalized_df = pd.DataFrame(X_normalized, columns=features)
    # Fit the model
    clusters = cluster.fit_predict(X_normalized_df)
    # Plot
    n_clusters = len(np.unique(clusters))
    # Add the cluster labels to the dataframe
    X_normalized_df['Cluster'] = clusters
    # Use pairplot to visualize each pair of features
    sns.pairplot(X_normalized_df, hue='Cluster', palette=sns.color_palette(n_colors=n_clusters))
    # Add features to the title
    str_features = ' - '
    for feature in features:
        str_features += feature + ' '
    plt.suptitle(title + str_features, y=1.02)
    plt.show()

    if features.__len__() == 3:
        if thd_plot: # 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X_normalized_df[features[0]], 
                           X_normalized_df[features[1]], 
                           X_normalized_df[features[2]], 
                           c=clusters.labels_, 
                           cmap='viridis')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                ax.set_title("3D - " + title + str_features)
                plt.show()
        if thd_view: # 3D plot with plotly
            fig = px.scatter_3d( X_normalized_df,
                x=features[0],
                y=features[1],
                z=features[2],
                color='Cluster',
                title="3D - " + title + str_features,
                labels={
                    features[0]: features[0],
                    features[1]: features[1],
                    features[2]: features[2],
                    'Cluster': 'Cluster ID'
                }
            )

            # Custom layout
            fig.update_layout(
                legend=dict(
                    title='Cluster ID',  # Title of the legend
                    x=1,  # Position orizontal
                    y=1  # Position vertical
                )
            )

            # Show the plot 3D
            fig.show()
    else:
        if thd_plot or thd_view:
            print('Features must be 3!')
    
# End