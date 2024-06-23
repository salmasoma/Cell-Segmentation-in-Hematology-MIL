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
import cv2
import numpy as np
import os
from albumentations import Compose, Resize, PadIfNeeded, LongestMaxSize, Sharpen, CLAHE
from PIL import Image
import torch
from torchvision import transforms
import torchstain
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

use_gpu = torch.cuda.is_available()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

worker_seed = torch.initial_seed() % 2**32
np.random.seed(worker_seed)
random.seed(worker_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = A.Compose(
    [
        A.LongestMaxSize(max_size=448),
        A.PadIfNeeded(
            min_height=448, 
            min_width=448, 
            border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Padding with black color for RGB images
        ),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def create_mask(image_dir, masks_dir):
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
    checkpoint_path = '/l/users/dawlat.akaila/deeplabv3_leukemia_AR.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove keys related to aux_classifier
    aux_keys = [key for key in checkpoint.keys() if "aux_classifier" in key]
    for key in aux_keys:
        del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                img_path = os.path.join(image_dir, filename)
                image = Image.open(img_path).convert("RGB")
                image = np.array(image)

                augmented = transform(image=image)
                input_tensor = augmented["image"].unsqueeze(0).to(device)

                output = model(input_tensor)['out']
                predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()

                # Convert the predicted mask to binary mask
                binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

                # Save the predicted mask
                mask_image = Image.fromarray(binary_mask)
                mask_image.save(os.path.join(masks_dir, filename))


target = cv2.cvtColor(cv2.imread("./ref2.jpg"), cv2.COLOR_BGR2RGB)
T = transforms.Compose([
transforms.ToTensor(),
transforms.Lambda(lambda x: x*255)
])
normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
normalizer.fit(T(target))
    
# Preprocessing
def normalization(img):
    to_transform = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t_to_transform = T(to_transform)
    norm = normalizer.normalize(I=t_to_transform)
    norm = np.array(norm).astype(np.uint8)
    return norm

def create_patches(image_folder, mask_folder, patches_dir):
    # Create the output folder if it does not exist
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)

    # Define the transformation: resize and pad to maintain aspect ratio
    transform = Compose([
        LongestMaxSize(max_size=448),
        PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        Sharpen(alpha=(0.1, 0.2), p=1),
        CLAHE(clip_limit=3.0, tile_grid_size=(2, 2), p=1),  # Local contrast enhancement
            ])

    transforms_resize = Compose([LongestMaxSize(max_size=448)])

    # Process each image and corresponding mask
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif','.tiff')):  # Adjust the extension based on your images
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)  # Assumes mask has same filename

            # Load image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and mask is not None:
                # Apply the transformations
                augmented = transform(image=image)
                image_resized = augmented['image']

                org_image = transforms_resize(image=image)['image']
            

                # normalization
                # normalized_image = normalization(image_resized)
                normalized_image = image_resized

                # Optional: Adjust mask processing here, e.g., apply threshold
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Find contours in the mask image
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cell_count = 0  # Initialize cell count
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 25 or h < 26:  # Skip small contours
                        continue
                
                    # Calculate square bounding box
                    side_len = max(w, h)
                    center_x, center_y = x + w // 2, y + h // 2
                    x_new = max(center_x - side_len // 2, 0)
                    y_new = max(center_y - side_len // 2, 0)
                
                
                    # Adjust coordinates if the square goes beyond the image dimensions
                    if x_new + side_len > normalized_image.shape[1]:
                        x_new = normalized_image.shape[1] - side_len
                    if y_new + side_len > normalized_image.shape[0]:
                        y_new = normalized_image.shape[0] - side_len
                                    
                    difference= (normalized_image.shape[0] - org_image.shape[0]) // 2
                    if difference >= y_new:
                        y_new = difference
                    elif y_new + side_len >= normalized_image.shape[0] - difference:
                        var_1 = (y_new + side_len) - (normalized_image.shape[0] - difference)
                        y_new = y_new - var_1

                    cropped_img = normalized_image[y_new:y_new + side_len, x_new:x_new + side_len]

                    # Resize the cell to 100x100
                    resized_img = cv2.resize(cropped_img, (100, 100))
                    resized_img = normalization(resized_img)
                    pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

                    # Save each cell with a unique filename
                    cell_count += 1
                    cell_filename = f"{os.path.splitext(filename)[0]}.{cell_count}.png"
                    cell_output_path = os.path.join(patches_dir, cell_filename)
                    pil_img.save(cell_output_path)

from collections import defaultdict

# Determine subgroups based on the given logic
def determine_subgroup(count):
    if count <= 10:
        return [1] * count
    else:
        subgroups = [(i // 10) + 1 for i in range(count)]
        # Handle the remainder
        remainder = count % 10
        if remainder != 0:
            last_full_group = (count // 10) - 1
            for i in range(remainder):
                subgroups[-(i + 1)] = last_full_group + 1
        return subgroups


# Create DF for MIL
def create_csv(directory, output_file='patch_test.csv', entry_type="test"):
    label_encodings = {"ALL": 4, "AML": 1, "CLL": 0, "CML": 3, "NORMAL": 2}
    current_label = 0

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Location', 'Subtype', 'Patient ID', 'label', 'train/test'])
        
        for dirpath, _, filenames in os.walk(directory):
            subtype = os.path.basename(dirpath)
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    location = os.path.join(dirpath, filename)
                    patient_id = "P1"
                    if subtype not in label_encodings:
                        label_encodings[subtype] = current_label
                        current_label += 1
                    label = label_encodings[subtype]
                    writer.writerow([filename, location, subtype, patient_id, label, entry_type])

    # Load CSV file
    df = pd.read_csv(output_file)
    df = df.sort_values(by='Filename').reset_index(drop=True)

    # Add initial subgroup assignments
    patient_id = "P1"
    count = len(df)
    subgroup_numbers = determine_subgroup(count)
    df['Patient ID_Subgroup'] = [f"{patient_id}_{subgroup}" for subgroup in subgroup_numbers]

    #rename 'Patient ID' column
    df.rename(columns={'Patient ID': 'Patient ID_Original'}, inplace=True)
    df.rename(columns={'Patient ID_Subgroup': 'Patient ID'}, inplace=True)

    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)

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
    patient_predictions = {}

    graph_net.eval()

    for batch_idx, (patient_ID, graph_object) in enumerate(test_loader.dataset.items()):

        data, label = graph_object

        with torch.no_grad():
            if use_gpu:
                data, label = data.to(device), label.to(device)
            else:
                data, label = data, label

        logits, Y_prob = graph_net(data.to(device))
        Y_hat = Y_prob.argmax(dim=1)

        patient_ID = patient_ID.split('_')[0]
        if patient_ID not in patient_predictions:
            patient_predictions[patient_ID] = []
        patient_predictions[patient_ID].append(Y_hat.item())

        del data, logits, Y_prob, Y_hat
        gc.collect()

    for patient_ID, preds in patient_predictions.items():
        print(f'Prediction Count: CLL: {preds.count(0)} AML: {preds.count(1)} NORMAL: {preds.count(2)} CML: {preds.count(3)} ALL: {preds.count(4)}')
    # Calculate the most common prediction for each patient during testing
    for patient_ID, preds in patient_predictions.items():
        CLL = preds.count(0)
        AML = preds.count(1)
        NORMAL = preds.count(2)
        CML = preds.count(3)
        ALL = preds.count(4)
        
        subtype_index = [4,1,0,3,2]
        list_count = [ALL, AML, CLL, CML, NORMAL]
        max_count = max(list_count)

        return subtype_index[list_count.index(max_count)]    

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
    graph_net = GAT_SAGPool(1024, heads=5, pooling_ratio=0.7).to(device)
    loss_fn = nn.CrossEntropyLoss()
    graph_net.load_state_dict(torch.load("./checkpoint_86.pth"), strict=True)

    labels = test_graph_multi_wsi(graph_net, test_graph_loader, loss_fn, n_classes=n_classes)
    return labels


class_number_to_subtype = {4:"ALL", 1:"AML", 0:"CLL", 3:"CML", 2:"NORMAL"}

def main():
    parser = argparse.ArgumentParser(description="Multi-stain self-attention graph multiple instance learning for Whole Slide Image set classification at the patient level")

    # Command line arguments
    parser.add_argument("--data", type=str, default="./data", help="Dataset Path")
    args = parser.parse_args()

    # Create the masks
    create_mask(args.data, './masks/')
    # Create the patches
    create_patches(args.data, './masks/', './patches/')
    # Create the CSV
    create_csv('./patches/', 'patch_test.csv')
    # Inference with MIL to get CLASS
    label = inference_mil('patch_test.csv')
    # Aggregate the predictions
    subtype = class_number_to_subtype[label]
    print(f"The predicted class is: {subtype}")

    # Clean up the directories
    shutil.rmtree('./masks/')
    shutil.rmtree('./patches/')
    os.remove('patch_test.csv')
    os.remove('test_graph_dict_inference.pkl')
    os.remove('test_embedding_dict_inference.pkl')

if __name__ == "__main__":
    main()