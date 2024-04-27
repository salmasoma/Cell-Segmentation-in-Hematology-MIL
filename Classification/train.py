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

train_ds = ImageFolder('/Users/salma/Desktop/Leuk/data/overlayed_train_w', transform=train_transforms)
test_ds = ImageFolder('/Users/salma/Desktop/Leuk/data/overlayed_val_w/', transform=test_transforms)

print(dict(Counter(train_ds.targets)))
print(dict(Counter(test_ds.targets)))

print(train_ds.class_to_idx)
print(test_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = LitModel(num_classes=NUM_CLASSES)
logger = pl.loggers.CSVLogger("logs", name="experiment")
checkpoint_f1 = pl.callbacks.ModelCheckpoint(monitor= "val_f1" ,mode = "max", filename='best_f1')
checkpoint_acc = pl.callbacks.ModelCheckpoint(monitor= "val_acc" ,mode = "max", filename='best_acc')
trainer = pl.Trainer(accelerator='mps', max_epochs=300, logger=logger, callbacks=[checkpoint_f1,checkpoint_acc])
trainer.fit(model, train_loader, test_loader)

