import numpy as np
import torch.nn.functional as F
from model_ssl import LitModel
import pytorch_lightning as pl
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image

BATCH_SIZE = 64

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
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.1,hue=0.05,saturation=.1,contrast=.1),
    transforms.GaussianBlur(3),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomAffine(degrees=0,scale=(0.6,1.0))
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    SquarePad(),
    transforms.Resize((224,224)),
    ])

train_ds = ImageFolder('all_tval/train/', transform=train_transforms)
test_ds = ImageFolder('all_tval/test/', transform=test_transforms)

print(dict(Counter(train_ds.targets)))
print(dict(Counter(test_ds.targets)))

print(train_ds.class_to_idx)
print(test_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)

model = LitModel()
logger = pl.loggers.CSVLogger("logs", name="noval_square_flip_jit_blur_shar_1e4_adamw_r18")
checkpoint = pl.callbacks.ModelCheckpoint(monitor= "val_f1" ,mode = "max", filename='best')
trainer = pl.Trainer(gpus=1, max_epochs=200, logger=logger, callbacks=[checkpoint])#, auto_lr_find=True)
# trainer.tune(model, train_loader)
trainer.fit(model, train_loader, test_loader)
trainer.test(model, test_loader, ckpt_path='best')
