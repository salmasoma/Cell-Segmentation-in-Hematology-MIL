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

BATCH_SIZE = 32

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
    #transforms.ColorJitter(brightness=0.05,hue=0.05,saturation=0.05,contrast=0.05),
    #transforms.GaussianBlur(3),
    #transforms.RandomAdjustSharpness(2),
    transforms.RandomAffine(degrees=0,scale=(0.6,1.0))
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    SquarePad(),
    transforms.Resize((448,448)),
    ])

train_ds = ImageFolder('all_new/train/', transform=train_transforms)
test_ds = ImageFolder('all_new/val/', transform=test_transforms)

print(dict(Counter(train_ds.targets)))
print(dict(Counter(test_ds.targets)))

print(train_ds.class_to_idx)
print(test_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)

model = LitModel()
logger = pl.loggers.CSVLogger("logs", name="new_noval_five_square_flip_1e3_adamw_d121")
checkpoint = pl.callbacks.ModelCheckpoint(monitor= "val_f1" ,mode = "max", filename='best')
trainer = pl.Trainer(gpus=1, max_epochs=300, logger=logger, callbacks=[checkpoint])#, auto_lr_find=True)
# trainer.tune(model, train_loader)
trainer.fit(model, train_loader, test_loader)
trainer.test(model, test_loader, ckpt_path='best')

## THE "WINNING" TRANSFORMS

# train_transforms = transforms.Compose([
# 	transforms.ToTensor(),
# 	SquarePad(),
#     transforms.Resize((448,448)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ColorJitter(brightness=.1,hue=0.05,saturation=.1,contrast=.1),
#     transforms.GaussianBlur(3),
#     transforms.RandomAdjustSharpness(2),
#     transforms.RandomAffine(degrees=0,scale=(0.6,1.0))
#     ])
