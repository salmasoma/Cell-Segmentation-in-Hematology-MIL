import torch
from madgrad import MADGRAD
from torch.nn.functional import nll_loss,log_softmax
from torchvision.models import resnet18, resnet50, densenet121
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix
import matplotlib.pyplot as plt
from typing import List
from nltk.tree import Tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

class LitModel(pl.LightningModule):
    def __init__(self, num_classes=7):
        super().__init__()

        self.ncls = num_classes
        self.net = densenet121(pretrained=True)
        self.net.classifier = torch.nn.Linear(self.net.classifier.in_features,self.ncls)
        self.criterion1 = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.CrossEntropyLoss()

        self.learning_rate = 1e-3

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = self(x)
        loss = self.criterion1(pred, y)
        pred = torch.argmax(pred, 1)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        prob = self(x)
        loss = self.criterion2(prob,y)
        pred = torch.argmax(prob,1)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return pred, y, prob

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

