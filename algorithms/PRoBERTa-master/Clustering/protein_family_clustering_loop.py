import numpy as np
import pandas as pd
import random

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
 
from RobertaLoader import RobertaDataset


EMBEDDING_SIZE = 192
NUM_PCA_COMPONENTS = 20
NUM_KMEANS_CLUSTERS = 4000

USE_NULL_MODEL = False

tokenized_data_filepath = 'human_fam_data_reviewed.csv'
roberta_model = '/home/nambiar4/Clustering/roberta.base'
roberta_weights = '/home/nambiar4/Clustering/192_easy2_checkpoint_best.pt'


def filtered_protein_df():
    '''
    Returns the tokenized sequence and protein family data from the given file
    (filtering out families that only contain a single protein).
    '''
    feat_cols = ['Protein families', 'Tokenized Sequence']
    tokenized_seq_df = pd.DataFrame(pd.read_csv(tokenized_data_filepath), columns=feat_cols)
    family_sizes = tokenized_seq_df['Protein families'].value_counts()
    single_protein_families = family_sizes[family_sizes <= 1].index
    return tokenized_seq_df[~tokenized_seq_df['Protein families'].isin(single_protein_families)].reset_index(drop=True)

def roberta_embeddings(roberta_dataset):
    embeddings = pd.DataFrame(index=np.arange(len(roberta_dataset)), columns=np.arange(EMBEDDING_SIZE))
    for i in range(len(roberta_dataset)):
        roberta_embedding = np.array(roberta_dataset[i][0].cpu())
        embeddings.iloc[i, :] = np.sum(roberta_embedding, axis=0)
        #print(np.sum(np.sum(roberta_embedding, axis=0)))
    #dat = embeddings.values
    #scaler = StandardScaler()
    #scaler.fit(dat)
    #dat = scaler.transform(data)
    #embeddings = pd.DataFrame(data=dat, columns=np.arrange(EMBEDDING_SIZE))
    #print(embeddings[:3])
    return embeddings

def compute_pca_results(embeddings_df):
    pca = PCA(n_components=NUM_PCA_COMPONENTS)
    return pd.DataFrame(pca.fit_transform(embeddings_df))

def append_kmeans_predictions(tokenized_seq_df, pca_df):
    '''
    Uses k-means to predict clusters for each protein, and adds these
    predictions to the tokenized sequence dataframe in a new column.
    '''
    kmeans = KMeans(n_clusters=NUM_KMEANS_CLUSTERS)
    kmeans_clusters = kmeans.fit_predict(normalize(pca_df, axis=1))
    tokenized_seq_df['Cluster'] = kmeans_clusters
    return tokenized_seq_df

def append_null_model_predictions(tokenized_seq_df):
    '''
    Predicts clusters for each protein randomly, and adds these
    predictions to the tokenized sequence dataframe in a new column.
    '''
    random_clusters = [random.randint(0, NUM_KMEANS_CLUSTERS - 1) for i in range(len(tokenized_seq_df))]
    tokenized_seq_df['Cluster'] = random_clusters
    return tokenized_seq_df

def nmi_score(cluster_predictions_df):
    true_labels = cluster_predictions_df['Protein families']
    predicted_labels = cluster_predictions_df['Cluster']
    return normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

if __name__ == '__main__':
    tokenized_seq_df = filtered_protein_df()
    cluster_predictions_df = pd.DataFrame()

    if USE_NULL_MODEL:
        cluster_predictions_df = append_null_model_predictions(tokenized_seq_df)
    else:
        roberta_dataset = RobertaDataset(model=roberta_model, weights=roberta_weights, df=tokenized_seq_df, layers=[2, 3, 4, 5])
        embeddings_df = roberta_embeddings(roberta_dataset)
        pca_results_df = compute_pca_results(embeddings_df)
        for i in range(20):
            cluster_predictions_df = append_kmeans_predictions(tokenized_seq_df.copy(), pca_results_df)
            print(nmi_score(cluster_predictions_df))
