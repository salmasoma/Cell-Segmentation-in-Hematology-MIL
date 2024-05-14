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

# Transform for the inference dataset
infer_transforms = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
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

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, img_path


def create_mask(resized_dir, masks_dir):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # If you are working with an auxiliary classifier, you should also adjust it
    if model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    checkpoint = torch.load("my_checkpoint_448.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

    # Instantiate the inference dataset and dataloader
    infer_dataset = InferenceDataset(image_dir=resized_dir, transform=infer_transforms)
    infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

    model.eval()
    for idx, (x, img_path) in enumerate(infer_loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)['out'])
            preds = (preds > 0.5).float()
        preds = preds.cpu().numpy()
        for i in range(len(preds)):
            # We have batch size of 1, so we access 0-th dimension which gives us the predicted mask
            mask = preds[i].squeeze()
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            save_path = os.path.join(masks_dir, os.path.basename(img_path[i]).replace(".png", "_mask.png"))
            #check if the path exists
            if not os.path.exists(masks_dir):
                os.makedirs(masks_dir)
            mask_img.save(save_path)


def get_processed_mask(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image: all non-black pixels will be set to white (255)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Fill holes using morphological closing operation
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size if necessary
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    #smoothen the edges
    closed_image = cv2.GaussianBlur(closed_image, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)

    # Save the processed image
    cv2.imwrite(output_path, closed_image)

# Post-process the mask
def post_processing(masks_dir, processed_masks_dir):
    # Traverse the source directory and process each image
    for subdir, dirs, files in os.walk(masks_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct the output path
                relative_path = os.path.relpath(subdir, masks_dir)
                output_dir = os.path.join(processed_masks_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)
                
                # Process the image
                get_processed_mask(filepath, output_path)
    

# Create the patches 
def create_patches(resized_dir, processed_masks_dir, patches_dir):
    # Iterate over the images in the image directory
    for filename in os.listdir(resized_dir):
        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']):
            base_name = os.path.splitext(filename)[0]
            mask_filename = f"{base_name}.png" if base_name.endswith('_mask') else f"{base_name}_mask.png"
            image_path = os.path.join(resized_dir, filename)
            mask_path = os.path.join(processed_masks_dir, mask_filename)

            # Load the original image and mask
            original_image = cv2.imread(image_path)
            mask_image = cv2.imread(mask_path, 0)

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

                # Resize the cell to 232x232 without padding
                resized_img = cv2.resize(cropped_img, (100, 100))
                pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

                # Save each cell with a unique filename
                cell_count += 1
                cell_filename = f"{os.path.splitext(filename)[0]}_{cell_count}.png"
                os.makedirs(patches_dir, exist_ok=True)
                cell_output_path = os.path.join(patches_dir, cell_filename)
                pil_img.save(cell_output_path)

# Create DF for MIL
def create_csv(patches_dir, patches_csv='patch_test.csv'):
    # Dictionary to store label encodings
    label_encodings = {"ALL": 4, "AML": 1, "CLL": 0, "CML": 3, "NORMAL": 2}
    current_label = 0

    # Prepare to write to CSV
    with open(patches_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(['Filename', 'Location', 'Subtype', 'Patient ID', 'label'])
        
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
                    # Get or create label encoding for the subtype
                    if subtype not in label_encodings:
                        label_encodings[subtype] = current_label
                        current_label += 1
                    label = label_encodings[subtype]
                    # Write data to CSV
                    writer.writerow([filename, location, subtype, patient_id, label])

class Loaders:

    def train_test_ids(self, df, train_fraction, random_state, patient_id, label, subset=False, split_column='train/test'):
        """
        Splits the IDs based on 'train' or 'test' values in a specified column of the dataframe.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            train_fraction (float): The fraction of data to be used for training (not used here but retained for compatibility).
            random_state (int): The random state for reproducibility (not used here but retained for compatibility).
            patient_id (str): The column name containing patient IDs.
            label (str): The column name containing labels (not used here but retained for compatibility).
            subset (bool): Whether to create a subset for smaller experimental runs.
            split_column (str): The column name to decide train/test split.

        Returns:
            tuple: A tuple containing full file IDs, train IDs, and test IDs.
        """
        train_ids = df[df[split_column] == 'train'][patient_id].unique().tolist()
        test_ids = df[df[split_column] == 'test'][patient_id].unique().tolist()
        file_ids = sorted(set(train_ids + test_ids))
    
        if subset:
            train_subset_ids = random.sample(train_ids, 10)
            test_subset_ids = random.sample(test_ids, 5)
            return file_ids, train_subset_ids, test_subset_ids
    
        return file_ids, train_ids, test_ids

    def df_loader(self, df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False):
        """
        Loads the training and testing DataFrame subsets based on patient IDs.

        Args:
            df (pd.DataFrame): The full DataFrame.
            train_transform (callable): Transformations to apply to the training data.
            test_transform (callable): Transformations to apply to the testing data.
            train_ids (list): List of patient IDs for training.
            test_ids (list): List of patient IDs for testing.
            patient_id (str): The column name containing patient IDs.
            label (str): The label column name.
            subset (bool): If True, use only a subset of data (not used here but retained for compatibility).

        Returns:
            tuple: Training and testing DataFrame subsets.
        """
        train_subset = df[df[patient_id].isin(train_ids)].reset_index(drop=True)
        test_subset = df[df[patient_id].isin(test_ids)].reset_index(drop=True)
        return train_subset, test_subset

    def slides_dataloader(self, train_sub, test_sub, train_ids, test_ids, train_transform, test_transform, slide_batch, num_workers, shuffle, collate, label='Pathotype_binary', patient_id="Patient ID"):
        """
        Creates data loaders for the training and testing subsets.

        Args:
            train_sub (pd.DataFrame): Training subset DataFrame.
            test_sub (pd.DataFrame): Testing subset DataFrame.
            train_ids (list): List of patient IDs for training data loaders.
            test_ids (list): List of patient IDs for testing data loaders.
            train_transform (callable): Transformations for the training data.
            test_transform (callable): Transformations for the testing data.
            slide_batch (int): Batch size for the data loaders.
            num_workers (int): Number of workers for data loading.
            shuffle (bool): Whether to shuffle the data loaders.
            collate (callable): Collate function for the data loaders.
            label (str): Label column name.
            patient_id (str): Patient ID column name.

        Returns:
            tuple: Dictionaries of data loaders for training and testing subsets.
        """
        # TRAIN dict
        train_subsets = {}
        for file in train_ids:
            new_key = f'{file}'
            train_subset = train_sub[train_sub[patient_id] == file]
            train_subsets[new_key] = torch.utils.data.DataLoader(train_subset, batch_size=slide_batch, shuffle=shuffle, num_workers=num_workers, drop_last=False, collate_fn=collate)

        # TEST dict
        test_subsets = {}
        for file in test_ids:
            new_key = f'{file}'
            test_subset = test_sub[test_sub[patient_id] == file]
            test_subsets[new_key] = torch.utils.data.DataLoader(test_subset, batch_size=slide_batch, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate)

        return train_subsets, test_subsets

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
                    inputs, label = inputs.to(device), label.to(device)
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

            knn_graph = kneighbors_graph(patient_embedding.reshape(-1, 1), k, mode=mode, include_self=include_self)
            edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
            data = Data(x=patient_embedding, edge_index=edge_index)
        
            graph_dict[patient_ID] = [data.to('cpu'), label.to('cpu')]
    
    return graph_dict, embedding_dict

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

def test_graph_multi_wsi(graph_net, test_loader, loss_fn, n_classes=2):
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

        test_acc += torch.sum(Y_hat == label.data)
        test_count += 1

        loss = loss_fn(logits, label)
        test_loss += loss.item()

        labels.append(label.item())

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
    # Image transforms
    train_transform = transforms.Compose([
            transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.1),
            transforms.ColorJitter(contrast=0.1),
            transforms.ColorJitter(saturation=0.1),
            transforms.ColorJitter(hue=0.1)]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    seed = 42
    seed_everything(seed)
    K= 5
    num_workers = 4
    batch_size = 10
    label = 'label'
    patient_id = 'Patient ID'
    dataset_name = 'inference'
    n_classes= 5

    # Load the dataset
    df = pd.read_csv(patches_csv, header=0)
    df = df.dropna(subset=[label])

    # create k-NNG with VGG patch embedddings
    file_ids, train_ids, test_ids = Loaders().train_test_ids(df, 0, seed, patient_id, label, False)
    train_subset, test_subset = Loaders().df_loader(df, train_transform, test_transform, train_ids, test_ids, patient_id, label, subset=False)
    train_slides, test_slides = Loaders().slides_dataloader(train_subset, test_subset, train_ids, test_ids, train_transform, test_transform, slide_batch=10, num_workers=num_workers, shuffle=False, collate=collate_fn_none, label=label, patient_id=patient_id)
    embedding_net = VGG_embedding(embedding_vector_size=1024, n_classes=n_classes)

    if use_gpu:
        embedding_net.to(device)
    # Save k-NNG with VGG patch embedddings for future use
    slides_dict = {('train_graph_dict_', 'train_embedding_dict_') : train_slides ,
                    ('test_graph_dict_', 'test_embedding_dict_'): test_slides}
    for file_prefix, slides in slides_dict.items():
        graph_dict, embedding_dict = create_embeddings_graphs(embedding_net, slides, k=K, mode='connectivity', include_self=False)
        print(f"Started saving {file_prefix[0]} to file")
        with open(f"{file_prefix[0]}{dataset_name}.pkl", "wb") as file:
            pickle.dump(graph_dict, file)  # encode dict into Pickle
            print("Done writing graph dict into pickle file")
        print(f"Started saving {file_prefix[1]} to file")
        with open(f"{file_prefix[1]}{dataset_name}.pkl", "wb") as file:
            pickle.dump(embedding_dict, file)  # encode dict into Pickle
            print("Done writing embedding dict into pickle file")    
        
    with open(f"test_graph_dict_{dataset_name}.pkl", "rb") as test_file:
        # Load the dictionary from the file
        test_graph_dict = pickle.load(test_file)
    
    test_graph_loader = torch.utils.data.DataLoader(test_graph_dict, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    graph_net = GAT_SAGPool(1024, heads=2, pooling_ratio=0.7)
    loss_fn = nn.CrossEntropyLoss()

    # Load the model
    # unzip the checkpoint
    with zipfile.ZipFile("./mil_checkpoint.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    graph_net.load_state_dict(torch.load("./mil_checkpoint.pth"), strict=True)

    labels = test_graph_multi_wsi(graph_net, test_graph_loader, loss_fn, n_classes=n_classes)
    return labels

# take the labels list wicj contains the class number and return the most repeated class number
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
    # Aggregate the predictions
    class_number = aggregate_prediction(label)
    subtype = class_number_to_subtype[class_number]
    print(f"The predicted class is: {subtype}")

main()