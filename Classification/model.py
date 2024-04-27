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
        self.net = AutoModelForImageClassification.from_pretrained("JorgeGIT/finetuned-Leukemia-cell")
        self.net.classifier = torch.nn.Linear(self.net.classifier.in_features, num_classes)
        # self.net = densenet121(pretrained=True)
        # self.net.classifier = torch.nn.Linear(self.net.classifier.in_features,self.ncls)

        # self.net = resnet50(pretrained=True)
        # self.net.fc = torch.nn.Linear(self.net.fc.in_features,self.ncls)
        # self.net = ViT('B_16_imagenet1k', pretrained=True)
        # self.net.fc = torch.nn.Linear(self.net.fc.in_features, num_classes)

        # Initialize the Swin Transformer model
        # self.net = timm.create_model("vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", pretrained=True, num_classes=self.ncls, in_chans=3)
        # self.net = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=self.ncls, in_chans=3)
        # # Check the feature dimension after adaptive pooling
        # with torch.no_grad():
        #     # Mock input to infer feature dimension
        #     mock_input = torch.zeros(1, 3, 224, 224)
        #     features_dim = self.net.forward_features(mock_input).shape[1]

        # # Adjust the linear layer to match the feature dimension
        # self.net.head = torch.nn.Sequential(
        #     torch.nn.AdaptiveAvgPool2d((1, 1)),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(features_dim, num_classes)  # Adjust this line
        # )

        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.CrossEntropyLoss()

        self.learning_rate = 1e-3

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)
        loss = self.criterion1(pred.logits, y)
        pred = torch.argmax(pred.logits, 1)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        prob = self(x)
        loss = self.criterion2(prob.logits,y)
        pred = torch.argmax(prob.logits,1)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return pred, y, prob.logits

    def validation_epoch_end(self, validation_step_outputs):
        preds = []
        targets = []
        probs = []

        for outs in validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
            probs.append(outs[2])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()
        probs = torch.cat(probs).cpu()
         
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
    