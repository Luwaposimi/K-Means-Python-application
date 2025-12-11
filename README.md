# K-Means-Python-application
The goal of this project was to analyze and visualize pickup patterns of New York City (NYC) yellow taxis. By applying clustering techniques to geographic data, we aimed to identify high-demand pickup zones and provide actionable insights that could inform transportation planning, taxi fleet allocation, and urban mobility strategies.
#Libaries
Python 3.14, seaborn 0.13.2, matplotlib 3.10.8, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3 scipi 1.16.3

#Files
kmeans_nyc_taxi.py        Main Python script for downloading data, clustering, and visualization.  
| cluster_centers.csv  | Coordinates of the cluster centers identified by KMeans.                 |
| elbow_silhouette.png | Plot showing Elbow and Silhouette scores to determine optimal clusters.  |
| clusters_result.png  | Scatter plot of NYC taxi pickup clusters.                                |


#Testing 

Unit tests ensure:
Data sample is non-empty.
optimal_k is reasonable.
Cluster labels match the number of data points.
Silhouette score is acceptable.


#Dependencies
Python 3.8+
pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm, pyarrow, shapefile, requests
