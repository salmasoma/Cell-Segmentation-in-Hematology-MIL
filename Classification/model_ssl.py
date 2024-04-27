import torch
import torchvision
from madgrad import MADGRAD
from torch.nn.functional import nll_loss,log_softmax
from torchvision.models import resnet18, resnet50, densenet121
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, ConfusionMatrix
from sklearn.metrics import roc_auc_score
from typing import List


ncls = 7

def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load('tenpercent_resnet18.ckpt', map_location='cuda:0')

        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        self.net = load_model_weights(self.net, state_dict)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, ncls)

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
        # print(prob.shape)

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
        # print(preds.shape, targets.shape, probs.shape)
        print()
        print(Accuracy(task="multiclass", num_classes=ncls, top_k=1, average='none')(preds,targets))
        print(F1Score(task="multiclass", num_classes=ncls, top_k=1, average='none')(preds,targets))
        print(AUROC(task="multiclass", num_classes=ncls, average='none')(probs,targets))
        print()

        acc = Accuracy(task="multiclass", num_classes=ncls, top_k=1, average='macro')(preds,targets).item()
        f1 = F1Score(task="multiclass", num_classes=ncls, top_k=1, average='macro')(preds,targets).item()
        auc = AUROC(task="multiclass", num_classes=ncls, average='macro')(probs,targets).item()

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_nb):
        x, y = batch
        prob = self(x)
        pred = torch.argmax(prob,1)
        return pred, y, prob

    def test_epoch_end(self, validation_step_outputs):
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
        # print(preds.shape, targets.shape, probs.shape)

        confmat = ConfusionMatrix(task="multiclass", num_classes=ncls)

        print()
        print(Accuracy(task="multiclass", num_classes=ncls, top_k=1, average='none')(preds,targets))
        print(F1Score(task="multiclass", num_classes=ncls, top_k=1, average='none')(preds,targets))
        print(AUROC(task="multiclass", num_classes=ncls, average='none')(probs,targets))
        print()
        print(confmat(preds,targets))
        print()

        acc = Accuracy(task="multiclass", num_classes=ncls, top_k=1, average='macro')(preds,targets).item()
        f1 = F1Score(task="multiclass", num_classes=ncls, top_k=1, average='macro')(preds,targets).item()
        auc = AUROC(task="multiclass", num_classes=ncls, average='macro')(probs,targets).item()

        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auc', auc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # return MADGRAD(self.parameters(), lr=self.learning_rate)
