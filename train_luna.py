import os
import glob
import monai
import torch
import numpy as np 
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk
from statistics import mean
from torch.optim import Adam
from natsort import natsorted
import matplotlib.pyplot as plt
from transformers import SamModel 
import matplotlib.patches as patches
from transformers import SamProcessor
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
from monai.transforms import ResizeD
    
from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
)
           
# create an instance of the processor for image preprocessing
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))
        
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 512, 512] # if there is no mask in the array, set bbox to image size
    



class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transforms = transforms = Compose([
            
            # load .nii or .nii.gz files
            LoadImaged(keys=['img', 'label']),
            
            # add channel id to match PyTorch configurations
            EnsureChannelFirstd(keys=['img', 'label']),
            
            # reorient images for consistency and visualization
            Orientationd(keys=['img', 'label'], axcodes='RA'),
            
            # resample all training images to a fixed spacing
#             Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),
            
            # rescale image and label dimensions to 256x256 
            # CenterSpatialCropd(keys=['img', 'label'], roi_size=(256,256)),
            ResizeD(keys=['img', 'label'], spatial_size=(256,256), mode=("bilinear", "nearest")),
            
            # scale intensities to 0 and 255 to match the expected input intensity range
            ScaleIntensityd(keys=['img']),
            ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=1.0, 
                         b_min=0.0, b_max=255.0, clip=True), 
            ScaleIntensityd(keys=['label']),
            # ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255, 
            #              b_min=0.0, b_max=1.0, clip=True), 

#             SpatialPadd(keys=["img", "label"], spatial_size=(256,256))
#             RepeatChanneld(keys=['img'], repeats=3, allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'img': image_path, 'label': mask_path})

        # squeeze extra dimensions
        image = data_dict['img'].squeeze()
        ground_truth_mask = data_dict['label'].squeeze()

        # convert to int type for huggingface's models expected inputs
        image = image.astype(np.uint8)

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image))
        
        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)
        
        # get bounding box prompt (returns xmin, ymin, xmax, ymax)
        # in this dataset, the contours are -1 so we change them to 1 for label and 0 for background
        ground_truth_mask[ground_truth_mask < 0] = 1
        
        prompt = get_bounding_box(ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))

        return inputs
# Initialize dictionary for storing image and label paths
data_paths = {}
datasets = ['train', 'val', 'test']
data_types = ['2d_images', '2d_masks']
# Create directories and print the number of images and masks in each
for dataset in datasets:
    for data_type in data_types:
        # Construct the directory path
        dir_path = os.path.join('./LUNA16/dataset_512/', f'{dataset}_{data_type}_512')
        
        # Find images and labels in the directory
        files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
        
        # Store the image and label paths in the dictionary
        data_paths[f'{dataset}_{data_type.split("_")[1]}'] = files

print('Number of training images', len(data_paths['train_images']))
print('Number of validation images', len(data_paths['val_images']))
print('Number of test images', len(data_paths['test_images']))
# create train and validation dataloaders
train_dataset = SAMDataset(image_paths=data_paths['train_images'], mask_paths=data_paths['train_masks'], processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = SAMDataset(image_paths=data_paths['val_images'], mask_paths=data_paths['val_masks'], processor=processor)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# example = train_dataset[5]
# for k,v in example.items():
#     print(k,v.shape)

# xmin, ymin, xmax, ymax = get_bounding_box(example['ground_truth_mask'])

# fig, axs = plt.subplots(1, 2)

# axs[0].imshow(example['pixel_values'][1], cmap='gray')
# axs[0].axis('off')

# axs[1].imshow(example['ground_truth_mask'], cmap='copper')

# # create a Rectangle patch for the bounding box
# rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

# # add the patch to the second Axes
# axs[1].add_patch(rect)

# axs[1].axis('off')

# plt.tight_layout()
# plt.show()
# load the pretrained weights for finetuning
model = SamModel.from_pretrained("facebook/sam-vit-large")

# make sure we only compute gradients for mask decoder (encoder weights are frozen)
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        print(name)
        param.requires_grad_(False)   
# define training loop
num_epochs = 200

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# define optimizer
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0.99)

# define segmentation loss with sigmoid activation applied to predictions from the model
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# track mean train and validation losses
mean_train_losses, mean_val_losses = [], []

# create an artibarily large starting validation loss value
best_val_loss = 1000.0
best_val_epoch = 0

# set model to train mode for gradient updating
model.train()
for epoch in range(num_epochs):
    
    # create temporary list to record training losses
    epoch_losses = []
    for i, batch in enumerate(tqdm(train_dataloader)):

        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())
        
        # # visualize training predictions every 50 iterations
        # if i % 50 == 0:
            
        #     # clear jupyter cell output
        #     clear_output(wait=True)
            
        #     fig, axs = plt.subplots(1, 3)
        #     xmin, ymin, xmax, ymax = get_bounding_box(batch['ground_truth_mask'][0])
        #     rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

        #     axs[0].set_title('input image')
        #     axs[0].imshow(batch["pixel_values"][0,1], cmap='gray')
        #     axs[0].axis('off')

        #     axs[1].set_title('ground truth mask')
        #     axs[1].imshow(batch['ground_truth_mask'][0], cmap='copper')
        #     axs[1].add_patch(rect)
        #     axs[1].axis('off')
            
        #     # apply sigmoid
        #     medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            
        #     # convert soft mask to hard mask
        #     medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
        #     medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        #     axs[2].set_title('predicted mask')
        #     axs[2].imshow(medsam_seg, cmap='copper')
        #     axs[2].axis('off')

        #     plt.tight_layout()
        #     plt.show()
    # if epoch % 2 == 0:
    # create temporary list to record validation losses
    val_losses = []
    
    # set model to eval mode for validation
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            
            # forward pass
            outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                    input_boxes=val_batch["input_boxes"].to(device),
                    multimask_output=False)
            
            # calculate val loss
            predicted_val_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))

            val_losses.append(val_loss.item())
        
        # # visualize the last validation prediction
        # fig, axs = plt.subplots(1, 3)
        # xmin, ymin, xmax, ymax = get_bounding_box(val_batch['ground_truth_mask'][0])
        # rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

        # axs[0].set_title('input image')
        # axs[0].imshow(val_batch["pixel_values"][0,1], cmap='gray')
        # axs[0].axis('off')

        # axs[1].set_title('ground truth mask')
        # axs[1].imshow(val_batch['ground_truth_mask'][0], cmap='copper')
        # axs[1].add_patch(rect)
        # axs[1].axis('off')

        # # apply sigmoid
        # medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

        # # convert soft mask to hard mask
        # medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
        # medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        # axs[2].set_title('predicted mask')
        # axs[2].imshow(medsam_seg, cmap='copper')
        # axs[2].axis('off')

        # plt.tight_layout()
        # plt.show()

        # save the best weights and record the best performing epoch
        if mean(val_losses) < best_val_loss:
            torch.save(model.state_dict(), f"best_luna_v1.pth")
            print(f"Model Was Saved! Current Best val loss {best_val_loss}")
            best_val_loss = mean(val_losses)
            best_val_epoch = epoch
        else:
            print("Model Was Not Saved!")
    
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    print(f'Mean val loss: {mean(val_losses)}')
    
        # mean_train_losses.append(mean(epoch_losses))
        # mean_val_losses.append(mean(val_losses))