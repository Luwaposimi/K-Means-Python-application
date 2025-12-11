import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import requests
from tqdm import tqdm
import pyarrow.parquet as pq
import shapefile  # pip install pyshp

warnings.filterwarnings("ignore")


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FULL_TRIP_FILE = os.path.join(DATA_DIR, "yellow_tripdata_2023-01.parquet")
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

SHAPE_ZIP = os.path.join(DATA_DIR, "taxi_zones.zip")
SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

COORD_FILE = os.path.join(DATA_DIR, "taxi_zones_centroids.csv")

ELBOW_PLOT = "elbow_silhouette.png"
CLUSTER_PLOT = "clusters_result.png"

sample_size = 10000



def download_with_progress(url: str, filename: str):
    if os.path.exists(filename):
        print(f"Already exists: {os.path.basename(filename)}")
        return
    print(f"Downloading {os.path.basename(filename)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024
    with open(filename, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))
    print("Downloaded successfully")

def validate_parquet(filename: str):
    try:
        pq.read_metadata(filename)
        print("Parquet file validated")
    except:
        print("Invalid Parquet, removing and re-downloading...")
        os.remove(filename)
        raise

def extract_centroids():
    if os.path.exists(COORD_FILE):
        print(f"Already extracted: {os.path.basename(COORD_FILE)}")
        return
    print("Extracting zone centroids...")
    shp_path = os.path.join(DATA_DIR, "taxi_zones.shp")
    if not os.path.exists(shp_path):
        with zipfile.ZipFile(SHAPE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    sf = shapefile.Reader(shp_path)
    records = []
    for sr in sf.shapeRecords():
        rec = sr.record
        shape = sr.shape
        location_id = rec[0]
        lon = (shape.bbox[0] + shape.bbox[2]) / 2
        lat = (shape.bbox[1] + shape.bbox[3]) / 2
        records.append((location_id, lon, lat))
    df_centroids = pd.DataFrame(records, columns=["LocationID", "longitude", "latitude"])
    df_centroids.to_csv(COORD_FILE, index=False)
    print(f"Extracted {len(df_centroids)} centroids")


download_with_progress(TRIP_URL, FULL_TRIP_FILE)
validate_parquet(FULL_TRIP_FILE)
download_with_progress(SHAPE_URL, SHAPE_ZIP)
extract_centroids()

print(f"\nLoading small sample of {sample_size:,} trips...")
df_trips = pd.read_parquet(FULL_TRIP_FILE, columns=['PULocationID']).sample(n=sample_size, random_state=42)
df_coords = pd.read_csv(COORD_FILE)

df = df_trips.merge(df_coords[['LocationID', 'longitude', 'latitude']],
                    left_on='PULocationID', right_on='LocationID', how='left')
df = df.dropna(subset=['longitude', 'latitude']).reset_index(drop=True)
print(f"Valid trips in sample: {len(df):,}")

X = df[['longitude', 'latitude']].values


print("\nRunning Elbow & Silhouette (super fast)...")
inertias = []
silhouettes = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(K_range, inertias, marker='o', color='tab:blue', label='Inertia')
ax1.set_ylabel('Inertia', color='tab:blue')
ax2 = ax1.twinx()
ax2.plot(K_range, silhouettes, marker='s', color='tab:red', label='Silhouette')
ax2.set_ylabel('Silhouette Score', color='tab:red')
plt.title('Elbow Method + Silhouette Score')
plt.xlabel('Number of clusters (k)')
fig.tight_layout()
plt.savefig(ELBOW_PLOT, dpi=200, bbox_inches='tight')
plt.close()
print(f"→ Saved {ELBOW_PLOT}")

optimal_k = 8

# ====================== FINAL CLUSTERING (instant) ======================
print(f"\nClustering with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(12, 10))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='cluster',
                palette='tab10', s=20, alpha=0.8, edgecolor=None, legend='full')
plt.title(f'NYC Yellow Taxi Pickup Clusters (k={optimal_k}) – Small Sample')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(CLUSTER_PLOT, dpi=300, bbox_inches='tight')
plt.close()
print(f"→ Saved {CLUSTER_PLOT}")

centers = pd.DataFrame(kmeans.cluster_centers_, columns=['center_lon', 'center_lat'])
centers.to_csv("cluster_centers.csv", index_label="cluster")
print("→ Saved cluster_centers.csv")

print("\nDone in seconds!")
print(f"   • {os.path.abspath(ELBOW_PLOT)}")
print(f"   • {os.path.abspath(CLUSTER_PLOT)}")