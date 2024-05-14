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




# Preprocessing 
def normalization(DIR):
    OUT = './color_normalized/'

    #if OUT directory does not exist, create it
    if not os.path.exists(OUT):
        os.makedirs(OUT)

    target = cv2.cvtColor(cv2.imread("./Image_4184.png"), cv2.COLOR_BGR2RGB)

    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
    ])

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))

    # Walk through all the subdirectories in DIR
    for subdir, dirs, files in os.walk(DIR):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(subdir, filename)
                to_transform = cv2.cvtColor(cv2.imread(os.path.join(filepath)), cv2.COLOR_BGR2RGB)
                t_to_transform = T(to_transform)
                norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
                #save H image
                H = np.array(H)
                E = np.array(E)
                norm = np.array(norm)
                file_noext = os.path.splitext(filename)[0]
                s = os.path.relpath(subdir, DIR)
                out_path = os.path.join(OUT, s)
                os.makedirs(out_path, exist_ok=True)  # Create output subdir if not exists
                cv2.imwrite(os.path.join(out_path, file_noext) + ".png", norm)

def resize_images(root_dir, output_dir, size=(448, 448)):
    for subdir, dirs, files in os.walk(root_dir):
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
                out_path = os.path.join(output_dir, os.path.relpath(subdir, root_dir))
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

# Transform for the inference dataset
infer_transforms = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

def create_mask(output_dir, save_folder):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # If you are working with an auxiliary classifier, you should also adjust it
    if model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    checkpoint = ""
    model.load_state_dict(checkpoint["state_dict"])

    # Instantiate the inference dataset and dataloader
    infer_dataset = InferenceDataset(image_dir=output_dir, transform=infer_transforms)
    infer_loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            save_path = os.path.join(save_folder, os.path.basename(img_path[i]).replace(".png", "_mask.png"))
            #check if the path exists
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            mask_img.save(save_path)

# Post-process the mask
def post_processing():

# Create the patches 
# Create DF for MIL
# Inference with MIL to get CLASS