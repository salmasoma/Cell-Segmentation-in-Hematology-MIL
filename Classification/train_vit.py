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

class LitModel(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ncls = num_classes
        # Load a pre-trained model
        self.net = AutoModelForImageClassification.from_pretrained("JorgeGIT/finetuned-Leukemia-cell")
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        return self.net(x)
    
    def preprocess_with_mask(self, x, mask):
        # Assuming x and mask are torch Tensors and mask is binary [0, 1]
        # x shape: [batch_size, channels, height, width]
        # mask shape: [batch_size, 1, height, width]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # Expand mask to have the same number of channels as x
        mask = mask.expand_as(x)
        # Apply mask
        return x * mask

    def training_step(self, batch, batch_nb):
        x, y, mask = batch
        x = self.preprocess_with_mask(x, mask)
        logits = self.forward(x).logits
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y, mask = batch
        x = self.preprocess_with_mask(x, mask)
        logits = self.forward(x).logits
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'preds': preds, 'targets': y, 'logits': logits}

    def validation_epoch_end(self, validation_step_outputs):
        preds = []
        targets = []
        probs = []

        for outs in validation_step_outputs:
            preds.append(outs['preds'])
            targets.append(outs['targets'])
            probs.append(outs['logits'])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()
        probs = torch.cat(probs).cpu()

        # Compute metrics here as needed
        print()
        print(Accuracy(task="multiclass", num_classes=self.ncls, top_k=1, average='none')(preds,targets))
        print(F1Score(task="multiclass", num_classes=self.ncls, top_k=1, average='none')(preds,targets))
        print(AUROC(task="multiclass", num_classes=self.ncls, average='none')(probs,targets))
        print()

        acc = Accuracy(task="multiclass", num_classes=self.ncls, top_k=1, average='macro')(preds,targets).item()
        f1 = F1Score(task="multiclass", num_classes=self.ncls, top_k=1, average='macro')(preds,targets).item()
        auc = AUROC(task="multiclass", num_classes=self.ncls, average='macro')(probs,targets).item()

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

from torchvision.datasets import DatasetFolder
from PIL import Image
import os
from torchvision import transforms    
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

class ImageFolderWithMasks(DatasetFolder):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, target_transform=None):
        super().__init__(img_dir, loader=Image.open, extensions=('png', 'jpg', 'jpeg'), transform=transform, target_transform=target_transform)
        self.mask_dir = mask_dir
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Extract the relative path for the image and create the corresponding mask path
        rel_path = os.path.relpath(path, self.root)
        mask_path = os.path.join(self.mask_dir, os.path.dirname(rel_path), os.path.basename(rel_path).replace(".jpg", ".png").replace(".png", "_mask.png").replace(".jpeg", "_mask.png"))

        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, mask

BATCH_SIZE = 32
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


# Define the mask transforms
mask_transforms = transforms.Compose([
    transforms.Resize((448, 448)),  # Resize to match the image dimensions
    transforms.ToTensor(),
])

# Update your data loaders to use the new dataset class
train_ds = ImageFolderWithMasks('./data/train', './data/generated_train_masks', transform=train_transforms, mask_transform=mask_transforms)
test_ds = ImageFolderWithMasks('./data/val', './data/val_norm_mask_resized', transform=test_transforms, mask_transform=mask_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = LitModel(num_classes=NUM_CLASSES)
logger = pl.loggers.CSVLogger("log", name="experiment")
checkpoint_f = pl.callbacks.ModelCheckpoint(monitor= "val_f1" ,mode = "max", filename='best_f1')
checkpoint_a = pl.callbacks.ModelCheckpoint(monitor= "val_acc" ,mode = "max", filename='best_Acc')

trainer = pl.Trainer(accelerator='cuda', max_epochs=300, logger=logger, callbacks=[checkpoint_f, checkpoint_a])
trainer.fit(model, train_loader, test_loader)