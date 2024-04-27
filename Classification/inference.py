import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForImageClassification
import torch
from madgrad import MADGRAD
from torch.nn.functional import nll_loss,log_softmax
from torchvision.models import resnet18, resnet50, densenet121
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix, Recall, Precision
import matplotlib.pyplot as plt
from typing import List
from nltk.tree import Tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import timm
from pytorch_pretrained_vit import ViT
from transformers import AutoModelForImageClassification
import numpy as np
import torch.nn.functional as F
from model import LitModel
import pytorch_lightning as pl
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

BATCH_SIZE = 16
NUM_CLASSES = 5

class SquarePad:
	def __call__(self, image):
		w, h = image.size()[-2:]
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (vp,vp,hp,hp)
		return F.pad(image, padding, 'constant')

train_transforms = transforms.Compose([
	transforms.ToTensor(),
	SquarePad(),
    transforms.Resize((448,448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.05,hue=0.05,saturation=0.05,contrast=0.05),
    transforms.GaussianBlur(3),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomAffine(degrees=0,scale=(0.6,1.0))
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    SquarePad(),
    transforms.Resize((448,448)),
    ])

train_ds = ImageFolder('./data/train', transform=train_transforms)
test_ds = ImageFolder('/Users/salma/Desktop/val', transform=test_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to('mps')
            targets = targets.to('mps')
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    accuracy = accuracy_score(all_targets, all_preds)

    return precision, recall, f1, accuracy

# Load the best model based on F1 score
model_path = '/Users/salma/Downloads/best_acc_vit.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=model_path, num_classes=NUM_CLASSES)
model.to('mps')
# Evaluate the model
precision, recall, f1, accuracy = evaluate_model(model, test_loader)

# Print the metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
