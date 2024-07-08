# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:14:46 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np

import torch

from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

use_gpu = torch.cuda.is_available()
# use_gpu = False
# if use_gpu:
#     print("Using CUDA")
    
import gc 
gc.enable()


def create_embeddings_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False):

    graph_dict = dict()
    embedding_dict = dict()

    embedding_net.eval()
    with torch.no_grad():

        for patient_ID, slide_loader in loader.items():
            patient_embedding = []
    
            for patch in slide_loader:
                inputs, label = patch
                label = label[0].unsqueeze(0)
                
                if use_gpu:
                    inputs, label = inputs.cuda(), label.cuda()
                else:
                    inputs, label = inputs, label
    
                embedding = embedding_net(inputs)
                embedding = embedding.to('cpu')
                embedding = embedding.squeeze(0).squeeze(0)
                patient_embedding.append(embedding)
    
            try:
                patient_embedding = torch.cat(patient_embedding)
            except RuntimeError:
                continue
            
            embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu')]

            knn_graph = kneighbors_graph(patient_embedding.reshape(-1,1), k, mode=mode, include_self=include_self)
            edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index)
        
            graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]
    
    return graph_dict, embedding_dict

# def create_embeddings_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False, use_gpu=True):
#     graph_dict = dict()
#     embedding_dict = dict()

#     device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
#     embedding_net.to(device).eval()

#     with torch.no_grad():
#         for patient_ID, slide_loader in loader.items():
#             embeddings = []
#             labels = []

#             for inputs, label in slide_loader:
#                 inputs = inputs.to(device)
#                 embedding = embedding_net(inputs)
#                 embeddings.append(embedding.squeeze().to('cpu'))  # Move to CPU after processing all patches if possible
#                 labels.append(label.to('cpu'))  # Collect labels if necessary

#             if not embeddings:
#                 print(f"No embeddings for patient {patient_ID}. Skipping.")
#                 continue

#             embeddings = torch.stack(embeddings)  # Ensure this operation is outside the loop
#             last_label = labels[-1]

#             embedding_dict[patient_ID] = [embeddings, last_label]

#             if embeddings.size(0) > 1:  # KNN graph only if there are multiple embeddings
#                 knn_graph = kneighbors_graph(embeddings.numpy(), k, mode=mode, include_self=include_self)
#                 edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
#                 data = Data(x=embeddings, edge_index=edge_index)
#             else:
#                 edge_index = torch.empty((2, 0), dtype=torch.long)
#                 data = Data(x=embeddings, edge_index=edge_index)

#             graph_dict[patient_ID] = [data, last_label]

#     return graph_dict, embedding_dict

