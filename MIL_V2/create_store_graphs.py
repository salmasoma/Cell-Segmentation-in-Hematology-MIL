# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:14:46 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np

import torch
import random
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

### Duplicating or randomly selecting values to ensure embedding length is 43
# def create_embeddings_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False):
#     graph_dict = dict()
#     embedding_dict = dict()

#     embedding_net.eval()
#     with torch.no_grad():
#         for patient_ID, slide_loader in loader.items():
#             patient_embedding = []

#             for patch in slide_loader:
#                 inputs, label = patch
#                 label = label[0].unsqueeze(0)
                
#                 if use_gpu:
#                     inputs, label = inputs.cuda(), label.cuda()
#                 else:
#                     inputs, label = inputs, label

#                 embedding = embedding_net(inputs)
#                 embedding = embedding.to('cpu')
#                 embedding = embedding.squeeze(0).squeeze(0)
#                 patient_embedding.append(embedding)

#             try:
#                 patient_embedding = torch.cat(patient_embedding)
#             except RuntimeError:
#                 continue
            
#             # Ensure the embedding length is 43, duplicating values if necessary or randomly selecting if more
#             embedding_length = patient_embedding.shape[0]
#             if embedding_length < 43:
#                 repeat_count = (43 + embedding_length - 1) // embedding_length 
#                 patient_embedding = patient_embedding.repeat(repeat_count)[:43]  
#             elif embedding_length > 43:
#                 random_indices = random.sample(range(embedding_length), 43)  # Randomly select 43 indices
#                 patient_embedding = patient_embedding[random_indices]
            
#             embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu')]

#             knn_graph = kneighbors_graph(patient_embedding.reshape(-1, 1), k, mode=mode, include_self=include_self)
#             edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
#             data = Data(x=patient_embedding, edge_index=edge_index)

#             graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]

#     return graph_dict, embedding_dict

