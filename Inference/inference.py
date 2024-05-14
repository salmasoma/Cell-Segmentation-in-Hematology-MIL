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

# Transform for the inference dataset
infer_transforms = A.Compose(
    [
        A.Resize(height=448, width=448),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

def create_mask(resized_dir, masks_dir):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # If you are working with an auxiliary classifier, you should also adjust it
    if model.aux_classifier is not None:
        model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    checkpoint = ""
    model.load_state_dict(checkpoint["state_dict"])

    # Instantiate the inference dataset and dataloader
    infer_dataset = InferenceDataset(image_dir=resized_dir, transform=infer_transforms)
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

# Inference with MIL to get CLASS
