import os
import cv2
import numpy as np
import torch
import torchstain
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import csv
import re 
import pandas as pd
import pickle
import random
import torch.nn as nn
from torchvision import models
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling
import torch.optim as optim
import zipfile
import gc
import argparse
from torchvision.models.segmentation import deeplabv3_resnet50
import warnings
warnings.filterwarnings("ignore")
import shutil

# Transform for the inference dataset
infer_transforms = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

device = torch.device("cpu")
use_gpu = torch.cuda.is_available()

# Preprocessing 
def normalization(input_dir):
    output_normalized_dir = './color_normalized/'

    #if OUT directory does not exist, create it
    if not os.path.exists(output_normalized_dir):
        os.makedirs(output_normalized_dir)

    target = cv2.cvtColor(cv2.imread("./Image_4184.png"), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))

    # Walk through all the subdirectories in DIR
    for subdir, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                filepath = os.path.join(subdir, filename)
                to_transform = cv2.cvtColor(cv2.imread(os.path.join(filepath)), cv2.COLOR_BGR2RGB)
                t_to_transform = T(to_transform)
                norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
                #save H image
                H = np.array(H)
                E = np.array(E)
                norm = np.array(norm)
                file_noext = os.path.splitext(filename)[0]
                s = os.path.relpath(subdir, input_dir)
                out_path = os.path.join(output_normalized_dir, s)
                os.makedirs(out_path, exist_ok=True)  # Create output subdir if not exists
                cv2.imwrite(os.path.join(out_path, file_noext) + ".png", norm)

def resize_images(output_normalized_dir, resized_dir, size=(448, 448)):
    for subdir, dirs, files in os.walk(output_normalized_dir):
        for filename in files:
            # Check for image file extensions
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                # Construct the full file path
                filepath = os.path.join(subdir, filename)
                # Read the image
                image = cv2.imread(filepath)
                # Resize the image
                resized_image = cv2.resize(image, size)
                # Construct the output path, maintaining the directory structure
                out_path = os.path.join(resized_dir, os.path.relpath(subdir, output_normalized_dir))
                # Make sure the directory exists
                os.makedirs(out_path, exist_ok=True)
                # Save the image
                cv2.imwrite(os.path.join(out_path, filename), resized_image)

def create_mask(resized_dir, masks_dir):
    # Ensure the masks directory exists
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    # Load model and modify the final layers
    model = deeplabv3_resnet50(pretrained=False)

    # Modify the last convolutional layer
    model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1))

    # If there's an auxiliary classifier, modify it similarly
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1))

    # Load the trained model weights
    checkpoint_path = './deeplabv3_leukemia.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove keys related to aux_classifier
    aux_keys = [key for key in checkpoint.keys() if "aux_classifier" in key]
    for key in aux_keys:
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # Transformation for the input images
    resize = 448
    transform = A.Compose(
        [
            A.Resize(height=resize, width=resize),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])

    # Traverse the subdirectories to find image files
    for root, _, files in os.walk(resized_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                img_path = os.path.join(root, filename)

                try:
                    image = Image.open(img_path).convert("RGB")
                    image = np.array(image)
                except Exception as e:
                    continue

                augmented = transform(image=image)
                input_tensor = augmented["image"].unsqueeze(0).to(device)

                try:
                    output = model(input_tensor)['out']
                    predicted_mask = torch.sigmoid(output).squeeze().detach().cpu().numpy()

                    # Convert the predicted mask to binary mask
                    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

                    # Create the corresponding subdirectory in masks_dir
                    relative_path = os.path.relpath(root, resized_dir)
                    mask_subdir = os.path.join(masks_dir, relative_path)
                    os.makedirs(mask_subdir, exist_ok=True)

                    # Save the predicted mask
                    mask_save_path = os.path.join(mask_subdir, filename)
                    mask_image = Image.fromarray(binary_mask)
                    mask_image.save(mask_save_path)
                except Exception as e:
                    continue


def get_processed_mask(image_path, output_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size if necessary
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # Smoothen the edges
    closed_image = cv2.GaussianBlur(closed_image, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

    # Save the processed image
    cv2.imwrite(output_path, closed_image)



def post_processing(masks_dir, processed_masks_dir):
    # Ensure the processed masks directory exists
    if not os.path.exists(processed_masks_dir):
        os.makedirs(processed_masks_dir)
    
    # Traverse the source directory and process each image
    for subdir, dirs, files in os.walk(masks_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                filepath = os.path.join(subdir, file)
                # Construct the output path
                relative_path = os.path.relpath(subdir, masks_dir)
                output_dir = os.path.join(processed_masks_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)
                # Process the image
                get_processed_mask(filepath, output_path)
    

def create_patches(resized_dir, processed_masks_dir, patches_dir):
    # Iterate over the images in the image directory
    for root, _, files in os.walk(resized_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
                base_name = os.path.splitext(filename)[0]
                mask_filename = f"{base_name}.png"  # Adjusted to match the new naming convention
                image_path = os.path.join(root, filename)
                mask_path = os.path.join(processed_masks_dir, os.path.relpath(root, resized_dir), mask_filename)

                # Load the original image and mask
                original_image = cv2.imread(image_path)
                mask_image = cv2.imread(mask_path, 0)

                # Check if the mask image exists
                if mask_image is None:
                    print(f"Warning: Mask not found for {image_path}, skipping.")
                    continue

                # Resize the original image and mask to 448x448
                original_image_resized = cv2.resize(original_image, (448, 448))

                # Optional: Adjust mask processing here, e.g., apply threshold
                _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

                # Find contours in the mask image
                contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cell_count = 0  # Initialize cell count
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Ignore contours with area less than 100 pixels
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_img = original_image_resized[y:y+h, x:x+w]

                    # Resize the cell to 100x100 without padding
                    resized_img = cv2.resize(cropped_img, (100, 100))
                    pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

                    # Save each cell with a unique filename
                    cell_count += 1
                    relative_path = os.path.relpath(root, resized_dir)
                    cell_output_dir = os.path.join(patches_dir, relative_path)
                    os.makedirs(cell_output_dir, exist_ok=True)
                    cell_filename = f"{base_name}_{cell_count}.png"
                    cell_output_path = os.path.join(cell_output_dir, cell_filename)
                    pil_img.save(cell_output_path)

from collections import defaultdict

# Create DF for MIL
def create_csv(patches_dir, patches_csv='patch_test.csv'):
    # Dictionary to store label encodings
    label_encodings = {"ALL": 4, "AML": 1, "CLL": 0, "CML": 3, "NORMAL": 2}
    current_label = 0
    
    # Dictionary to count occurrences of patient IDs
    patient_id_counts = defaultdict(int)

    # List to temporarily store rows
    rows = []

    # Walk through all files in the directory
    for dirpath, _, filenames in os.walk(patches_dir):
        subtype = os.path.basename(dirpath)
        for filename in filenames:
            # Check if the file is an image or relevant file
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Create full path
                location = os.path.join(dirpath, filename)
                # Remove extension and '_overlayed' from filename to get Patient ID
                patient_id = re.sub(r'_(\d+)(\.[\w\d]+)$', '', filename)  # Use regex to remove last underscore followed by a number and extension
                # Set constant PID
                # Make patient ID be everything before '_'
                patient_id = patient_id.split('_')[1]
                # patient_id = "P1"
                # Get or create label encoding for the subtype
                if subtype not in label_encodings:
                    label_encodings[subtype] = current_label
                    current_label += 1
                label = label_encodings[subtype]
                subtype = "t"
                train_test = 'test'
                
                # Prepare the row data
                row = [filename, location, subtype, patient_id, label, train_test]
                rows.append(row)
                patient_id_counts[patient_id] += 1
    
    # Duplicate rows for patient IDs that appear only once
    duplicated_rows = []
    for row in rows:
        patient_id = row[3]
        duplicated_rows.append(row)
        if patient_id_counts[patient_id] == 1:
            duplicated_rows.append(row)  # Duplicate the row

    # Write to CSV
    with open(patches_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(['Filename', 'Location', 'Subtype', 'Patient ID', 'label', 'train/test'])
        # Write all rows
        writer.writerows(duplicated_rows)

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.df.iloc[idx]

        image_path = sample['Location']
        label = sample['label']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image, torch.tensor(label)

# Modify the df_loader method in Loaders class
class Loaders:
    def train_test_ids(self, df, train_fraction, random_state, patient_id, label, subset=False, split_column='train/test'):
        train_ids = df[df[split_column] == 'train'][patient_id].unique().tolist()
        test_ids = df[df[split_column] == 'test'][patient_id].unique().tolist()
        file_ids = sorted(set(train_ids + test_ids))
    
        if subset:
            train_subset_ids = random.sample(train_ids, 10)
            test_subset_ids = random.sample(test_ids, 5)
            return file_ids, train_subset_ids, test_subset_ids
    
        return file_ids, train_ids, test_ids
    def df_loader(self, df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False):
        test_subset = df.reset_index(drop=True)
        return test_subset

    def slides_dataloader(self, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, collate, label='Pathotype_binary', patient_id="Patient ID"):
        # TEST dict
        test_subsets = {}
        for file in test_ids:
            new_key = f'{file}'
            test_subset = test_sub[test_sub[patient_id] == file]
            test_subsets[new_key] = DataLoader(CustomDataset(test_subset, transform=test_transform), batch_size=slide_batch, shuffle=False, num_workers=num_workers, collate_fn=collate)
        return test_subsets

def filter_invalid_edges(data):
    num_nodes = data.x.size(0) 
    valid_mask = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
    if not valid_mask.all():
        data.edge_index = data.edge_index[:, valid_mask]
        return data

# def create_embeddings_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False):
#     graph_dict = dict()
#     embedding_dict = dict()

#     embedding_net.eval()
#     with torch.no_grad():
#         for patient_ID, slide_loader in loader.items():
#             patient_embedding = []
#             for patch in slide_loader:
#                 try:
#                     inputs, label = patch
#                     label = label[0].unsqueeze(0)

#                     if use_gpu:
#                         inputs, label = inputs.to(device), label.to(device)
#                     else:
#                         inputs, label = inputs, label

#                     embedding = embedding_net(inputs)
#                     embedding = embedding.to('cpu')
#                     embedding = embedding.squeeze(0).squeeze(0)
#                     patient_embedding.append(embedding)
#                 except KeyError as e:
#                     print(f"KeyError: {e}")
#                     continue

#             try:
#                 patient_embedding = torch.cat(patient_embedding)
#             except RuntimeError:
#                 continue
            
#             embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu')]

#             knn_graph = kneighbors_graph(patient_embedding.reshape(-1, 1), k, mode=mode, include_self=include_self)
#             edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
#             data = Data(x=patient_embedding, edge_index=edge_index)
        
#             graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]

#     return graph_dict, embedding_dict

def create_embeddings_graphs(embedding_net, loader, k=5, mode='connectivity', include_self=False):
    graph_dict = dict()
    embedding_dict = dict()

    embedding_net.eval()
    with torch.no_grad():
        for patient_ID, slide_loader in loader.items():
            patient_embedding = []
            for patch in slide_loader:
                try:
                    inputs, label = patch
                    label = label[0].unsqueeze(0)

                    if use_gpu:
                        inputs, label = inputs.to(device), label.to(device)
                    else:
                        inputs, label = inputs, label

                    embedding = embedding_net(inputs)
                    embedding = embedding.to('cpu')
                    embedding = embedding.squeeze(0).squeeze(0)
                    patient_embedding.append(embedding)
                except KeyError as e:
                    print(f"KeyError: {e}")
                    continue

            try:
                patient_embedding = torch.cat(patient_embedding)
            except RuntimeError:
                continue
            
            embedding_dict[patient_ID] = [patient_embedding.to('cpu'), label.to('cpu')]

            knn_graph = kneighbors_graph(patient_embedding.reshape(-1, 1), k, mode=mode, include_self=include_self)
            edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index)
        
            # Here is where you should filter invalid edges
            data = filter_invalid_edges(data)

            graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]

    return graph_dict, embedding_dict

class VGG_embedding(nn.Module):

    """
    VGG16 embedding network for WSI patches
    """

    def __init__(self, embedding_vector_size=1024, n_classes=2):

        super(VGG_embedding, self).__init__()

        embedding_net = models.vgg16_bn(pretrained=True)

        # Freeze training for all layers
        for param in embedding_net.parameters():
            param.require_grad = False

        # Newly created modules have require_grad=True by default
        num_features = embedding_net.classifier[6].in_features
        features = list(embedding_net.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, embedding_vector_size)])
        features.extend([nn.Dropout(0.5)])
        features.extend([nn.Linear(embedding_vector_size, n_classes)]) # Add our layer with n outputs
        embedding_net.classifier = nn.Sequential(*features) # Replace the model classifier

        features = list(embedding_net.classifier.children())[:-2] # Remove last layer
        embedding_net.classifier = nn.Sequential(*features)
        self.vgg_embedding = nn.Sequential(embedding_net)

    def forward(self, x):

        output = self.vgg_embedding(x)
        output = output.view(output.size()[0], -1)
        return output


class GAT_SAGPool(torch.nn.Module):

    """Graph Attention Network for full slide graph"""

    def __init__(self, dim_in, heads=2, pooling_ratio=0.7):

        super().__init__()

        self.pooling_ratio = pooling_ratio
        self.heads = heads

        self.gat1 = GATv2Conv(dim_in, 512, heads=self.heads, concat=False)
        self.gat2 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat3 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat4 = GATv2Conv(512, 512, heads=self.heads, concat=False)

        self.topk1 = SAGPooling(512, pooling_ratio)
        self.topk2 = SAGPooling(512, pooling_ratio)
        self.topk3 = SAGPooling(512, pooling_ratio)
        self.topk4 = SAGPooling(512, pooling_ratio)

        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 5)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat4(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out


def test_graph_multi_wsi(graph_net, test_loader, loss_fn, n_classes=5):
    gc.enable()
    labels = []

    graph_net.eval()

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        data, label = graph_object

        with torch.no_grad():
            if use_gpu:
                data, label = data.to(device), label.to(device)
            else:
                data, label = data, label

        logits, Y_prob = graph_net(data)
        Y_hat = Y_prob.argmax(dim=1)

        loss = loss_fn(logits, label)

        labels.append(Y_hat.item())

        del data, logits, Y_prob, Y_hat
        gc.collect()
    return labels
    

# Define collate function
def collate_fn_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# Inference with MIL to get CLASS
def inference_mil(patches_csv):
    train_transform = A.Compose([
        A.OneOf([
            A.ColorJitter(brightness=0.1),
            A.ColorJitter(contrast=0.1),
            A.ColorJitter(saturation=0.1),
            A.ColorJitter(hue=0.1)], p=1.0),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    seed = 42
    seed_everything(seed)
    K = 5
    num_workers = 4
    batch_size = 10
    label = 'label'
    patient_id = 'Patient ID'
    dataset_name = 'inference'
    n_classes = 5

    df = pd.read_csv(patches_csv, header=0)
    df = df.dropna(subset=[label])

    loaders = Loaders()
    file_ids, train_ids, test_ids = loaders.train_test_ids(df, 0, seed, patient_id, label, False)
    test_subset = loaders.df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False)
    test_slides = loaders.slides_dataloader(test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=10, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
    embedding_net = VGG_embedding(embedding_vector_size=1024, n_classes=n_classes)
    if use_gpu:
        embedding_net.to(device)

    slides_dict = {('test_graph_dict_', 'test_embedding_dict_'): test_slides}
    for file_prefix, slides in slides_dict.items():
        graph_dict, embedding_dict = create_embeddings_graphs(embedding_net, slides, k=K, mode='connectivity', include_self=False)
        with open(f"{file_prefix[0]}{dataset_name}.pkl", "wb") as file:
            pickle.dump(graph_dict, file)
        with open(f"{file_prefix[1]}{dataset_name}.pkl", "wb") as file:
            pickle.dump(embedding_dict, file)

    with open(f"test_graph_dict_{dataset_name}.pkl", "rb") as test_file:
        test_graph_dict = pickle.load(test_file)

    test_graph_loader = DataLoader(test_graph_dict, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    graph_net = GAT_SAGPool(1024, heads=2, pooling_ratio=0.7)
    loss_fn = nn.CrossEntropyLoss()
    graph_net.load_state_dict(torch.load("./mil_leukemia.pth"), strict=True)

    labels = test_graph_multi_wsi(graph_net, test_graph_loader, loss_fn, n_classes=n_classes)
    return labels


# take the labels list which contains the class number and return the most repeated class number
def aggregate_prediction(labels):
    return max(set(labels), key = labels.count)

class_number_to_subtype = {4:"ALL", 1:"AML", 0:"CLL", 3:"CML", 2:"NORMAL"}

def main():
    parser = argparse.ArgumentParser(description="Multi-stain self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--data", type=str, default="./data", help="Dataset Path")
    args = parser.parse_args()

    # Normalize the images
    normalization(args.data)
    # Resize the images
    resize_images('./color_normalized/', './resized/')
    # Create the masks
    create_mask('./resized/', './masks/')
    # Post-process the masks
    post_processing('./masks/', './processed_masks/')
    # Create the patches
    create_patches('./resized/', './processed_masks/', './patches/')
    # Create the CSV
    create_csv('./patches/', 'patch_test.csv')
    # Inference with MIL to get CLASS
    label = inference_mil('patch_test.csv')
    # print("All labels:", label)
    # Aggregate the predictions
    class_number = aggregate_prediction(label)
    subtype = class_number_to_subtype[class_number]
    print(f"The predicted class is: {subtype}")

    # Clean up the directories
    shutil.rmtree('./color_normalized/')
    shutil.rmtree('./resized/')
    shutil.rmtree('./masks/')
    shutil.rmtree('./processed_masks/')
    shutil.rmtree('./patches/')
    os.remove('patch_test.csv')
    os.remove('test_graph_dict_inference.pkl')
    os.remove('test_embedding_dict_inference.pkl')

if __name__ == "__main__":
    main()