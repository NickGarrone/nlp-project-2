import os
import shutil
import PyPDF2
import docx
import nltk
import gensim
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import hdbscan
import openai
import requests
import re

# Configuration
MODEL_PATH = 'doc2vec_model.d2v'
UNSEEN_FOLDER_PATH = 'unseen_files'
OUTPUT_BASE_DIR = 'clustered_files'
MAX_DEPTH = 3

# HDBSCAN Configuration
# MIN_CLUSTER_SIZE = 7
# MIN_SAMPLES = 2

MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 1

# Document Loading Functions
def load_doc2vec_model(model_path):
    return gensim.models.Doc2Vec.load(model_path)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf_file(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return ''.join(page.extract_text() + '\n' for page in reader.pages)

def read_word_file(file_path):
    doc = docx.Document(file_path)
    return '\n'.join(para.text for para in doc.paragraphs)

def read_files_in_directory(directory_path):
    texts = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith('.txt'):
            texts[filename] = read_text_file(file_path)
        elif filename.endswith('.pdf'):
            texts[filename] = read_pdf_file(file_path)
        elif filename.endswith('.docx'):
            texts[filename] = read_word_file(file_path)
    return texts

# Vector Inference
def infer_vectors(model, unseen_docs):
    return [model.infer_vector(preprocess_text(content)) for content in unseen_docs.values()]

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# Clustering and Directory Management
def create_cluster_directories(unseen_docs, clusters, base_output_dir, folder_path):
    os.makedirs(base_output_dir, exist_ok=True)
    unique_clusters = set(clusters)

    for cluster in unique_clusters:
        if cluster != -1:
            cluster_files = [filename for filename, c in zip(unseen_docs.keys(), clusters) if c == cluster]
            cluster_dir = os.path.join(base_output_dir, f'cluster_{cluster}')
            os.makedirs(cluster_dir, exist_ok=True)

            for filename in cluster_files:
                src_path = os.path.join(folder_path, filename)
                dest_path = os.path.join(cluster_dir, filename)
                shutil.move(src_path, dest_path)


def generate_cluster_labels_with_chatgpt(unseen_docs, clusters):
    api_key = 'key-here'
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    cluster_files = {}
    
    for filename, cluster in zip(unseen_docs.keys(), clusters):
        if cluster != -1:  # Exclude noise
            if cluster not in cluster_files:
                cluster_files[cluster] = []
            cluster_files[cluster].append(filename)
    
    cluster_labels = {}
    for cluster, filenames in cluster_files.items():
        prompt = f"Generate a very concise and descriptive label under 5 words for a folder containing these files: {', '.join(filenames)}."
        data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'user', 'content': f"Generate a concise and descriptive label for a folder containing these files: {', '.join(filenames)}."}
        ],
        'max_tokens': 100, 
        'temperature': 0.7  
    }

        response = requests.post(url, headers=headers, json=data)
        result = response.json()  # Convert response to a dictionary
        label = result['choices'][0]['message']['content']
        label = label[:50]
        label = re.sub(r'[\/:*?"<>|]', '_', label)
        cluster_labels[cluster] = label

        print(label)

    return cluster_labels

def rename_cluster_directories(base_output_dir, cluster_labels):
    for cluster, label in cluster_labels.items():
        old_dir = os.path.join(base_output_dir, f'cluster_{cluster}')
        new_dir = os.path.join(base_output_dir, label)
        if not os.path.exists(new_dir):
            os.rename(old_dir, new_dir)

def assign_noise_to_nearest_cluster(document_vectors_scaled, clusters, cluster_centroids):
    noise_indices = np.where(clusters == -1)[0]
    
    if len(noise_indices) == 0:
        return  

    if len(cluster_centroids) == 0:
        return

    for noise_index in noise_indices:
        noise_vector = document_vectors_scaled[noise_index].reshape(1, -1)
        distances = pairwise_distances(noise_vector, cluster_centroids)
        nearest_cluster_index = np.argmin(distances)
        clusters[noise_index] = nearest_cluster_index

# Clustering Process with Recursive Structure
def cluster_and_organize_documents(model_path, folder_path, depth=0):
    model = load_doc2vec_model(model_path)
    unseen_docs = read_files_in_directory(folder_path)
    
    if not unseen_docs:
        return

    document_vectors = infer_vectors(model, unseen_docs)
    
    # Standardize the vectors
    scaler = StandardScaler()
    document_vectors_scaled = scaler.fit_transform(document_vectors)

    # Apply PCA to reduce dimensionality for recursive clustering
    target_dim = 5 if depth > 0 else document_vectors_scaled.shape[1] # Keep original dimension at top level 
    if depth > 0:
        target_dim = min(target_dim, len(document_vectors_scaled))
    pca = PCA(n_components=target_dim)
    document_vectors_scaled = pca.fit_transform(document_vectors_scaled) 
    min_cluster_size = max(2, len(unseen_docs) // 2) if depth > 0 else MIN_CLUSTER_SIZE
    min_samples = 1 if depth == 0 else max(1, min_cluster_size // 2)

    # Run HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
    clusters = hdbscan_clusterer.fit_predict(document_vectors_scaled)

    print(f"Folder path: {folder_path}")
    print(f"Depth: {depth}")
    print(f"Clusters: {clusters}")

    # Evaluate the clusters
    if len(set(clusters)) > 1: #and -1 in set(clusters): 
        silhouette = silhouette_score(document_vectors_scaled, clusters)
        db_index = davies_bouldin_score(document_vectors_scaled, clusters)
        print(f'Silhouette Score: {silhouette:.3f}')
        print(f'Davies-Bouldin Index: {db_index:.3f}')
    else:
        print("Insufficient clusters for evaluation.")
        return

    # Visualize with PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(document_vectors_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title("HDBSCAN Clusters Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Calculate cluster centroids
    cluster_centroids = []
    for cluster in set(clusters):
        if cluster != -1:
            cluster_centroids.append(np.mean(document_vectors_scaled[clusters == cluster], axis=0))
    cluster_centroids = np.array(cluster_centroids)

    # Reassign noise points to the nearest cluster
    assign_noise_to_nearest_cluster(document_vectors_scaled, clusters, cluster_centroids)

    # Create directories and organize files based on updated cluster assignments
    output_directory = os.path.join(OUTPUT_BASE_DIR, os.path.relpath(folder_path, start=UNSEEN_FOLDER_PATH))
    os.makedirs(output_directory, exist_ok=True)
    create_cluster_directories(unseen_docs, clusters, output_directory, folder_path)

    # Generate cluster labels and rename directories using ChatGPT
    cluster_labels = generate_cluster_labels_with_chatgpt(unseen_docs, clusters)
    rename_cluster_directories(output_directory, cluster_labels)

    # Recursively cluster within each subdirectory if not at max depth
    if depth < MAX_DEPTH:
        for item in os.listdir(output_directory):
            item_path = os.path.join(output_directory, item)
            if os.path.isdir(item_path):
                cluster_and_organize_documents(model_path, item_path, depth + 1)

if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    cluster_and_organize_documents(MODEL_PATH, UNSEEN_FOLDER_PATH)
