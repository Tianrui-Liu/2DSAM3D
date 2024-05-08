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
#             CenterSpatialCropd(keys=['img', 'label'], roi_size=(256,256)),
            ResizeD(keys=['img', 'label'], spatial_size=(256,256), mode=("lanczos", "bilinear")),
            ScaleIntensityd(keys=['img']),
            ScaleIntensityRanged(keys=['img'], a_min=0.0, a_max=1.0, 
                         b_min=0.0, b_max=255.0, clip=True), 
            ScaleIntensityd(keys=['label']),

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
        rotated = np.rot90(image, -1)

            # 左右翻转
        flipped = np.fliplr(rotated)

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((flipped, flipped, flipped))
        # convert the grayscale array to RGB (3 channels)
        # array_rgb = np.dstack((image, image, image))
        
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
        inputs["image_name"] = os.path.basename(image_path).replace('.nii.gz', '.png') # Change the extension

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
# train_dataset = SAMDataset(image_paths=data_paths['train_images'], mask_paths=data_paths['train_masks'], processor=processor)
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# val_dataset = SAMDataset(image_paths=data_paths['val_images'], mask_paths=data_paths['val_masks'], processor=processor)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
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
# create test dataloader
test_dataset = SAMDataset(image_paths=data_paths['test_images'], mask_paths=data_paths['test_masks'], processor=processor)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
import numpy as np
from scipy.ndimage import label, binary_dilation, sum as ndi_sum
import cv2

device = "cuda:1" if torch.cuda.is_available() else "cpu"
state_dict = torch.load("best_weights_l_v2.pth")
model.load_state_dict(state_dict)
model.to(device)
if not os.path.exists('./SAM_label/'):
    os.makedirs('./SAM_label/')
with torch.no_grad():
    # cnt=0
    for batch in tqdm(test_dataloader):

        result_image_name = os.path.join('./SAM_label/', batch["image_name"][0])
        if os.path.exists(result_image_name):
            continue
        # result_image = Image.fromarray((closed_seg).astype(np.uint8))
        # result_image.save(result_image_name)
        # cnt+=1
        # if cnt<30:
        #     continue
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].cuda(1),
                      input_boxes=batch["input_boxes"].cuda(1),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().cuda(1)

        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        # print(medsam_seg.shape)
        # print(medsam_seg.dtype)  
        
        # Remove regions connected to the image border
        labeled, num_features = label(medsam_seg)
        border_mask = np.zeros_like(medsam_seg)
        border_mask[:, 0] = 1
        border_mask[0, :] = 1
        border_mask[:, -1] = 1
        border_mask[-1, :] = 1

        touching_border = np.unique(labeled * binary_dilation(border_mask))
        for region in touching_border:
            if region != 0:
                medsam_seg[labeled == region] = 0
               
        # plt.imshow(batch["pixel_values"][0, 1], cmap='gray')
        # plt.show()
        # Visualize the image after preprocessing (before Canny)
        # plt.imshow(medsam_seg, cmap='copper')
        # plt.title('before Canny')
        # plt.show()
        # 使用Canny边缘检测
        original_image = batch["pixel_values"][0, 1].cpu().numpy().astype(np.uint8)

        edges_original = cv2.Canny(original_image, 50, 200)
        # plt.imshow(edges_original, cmap='gray')
        # plt.title('Canny Edges of Original Image')
        # plt.show()

        kernel = np.ones((3,3),np.uint8)
        # closing = cv2.morphologyEx(edges_original, cv2.MORPH_CLOSE, kernel)

        dilated = cv2.dilate(edges_original, kernel, iterations=1)
        # # 显示结果
        # plt.imshow(dilated,cmap='gray')
        # plt.show()
        # 找到所有的轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]


        # 创建一个全黑的图像
        mask = np.zeros_like(edges_original)
        # 在mask上画出最大的轮廓
        cv2.drawContours(mask, [max_contour],-1, (255), thickness=cv2.FILLED)

        # 创建一个比输入图像大2的掩码，用于floodFill函数
        h, w = mask.shape[:2]
        mask_floodfill = np.zeros((h+2, w+2), np.uint8)

        # floodFill函数会改变输入图像，所以我们使用它的副本
        mask_floodfill_copy = mask.copy()

        # 找到一个种子点
        seed_point = (w//2, h//2)

        # 执行floodFill函数，将与种子点连通的区域填充为白色
        cv2.floodFill(mask_floodfill_copy, mask_floodfill, seed_point, 255)

        # 反转floodfill后的结果
        # mask_floodfill_inv = cv2.bitwise_not(mask_floodfill_copy)

        # 结合原始的mask和floodfill的结果，得到最终的mask
        final_mask = mask | mask_floodfill_copy

        final_mask = cv2.resize(final_mask, (256, 256))
        # print(final_mask.shape)
        # print(final_mask.dtype)
        # channels = mask.shape[2] if len(mask.shape) == 3 else 1
        # print(channels)
        # channels = medsam_seg.shape[2] if len(mask.shape) == 3 else 1
        # print(channels)
        # print("--------")
        # Keep regions with area greater than 2*2
        for i in range(1, num_features + 1):
            area = ndi_sum(labeled == i)
            if area <= 4:
                medsam_seg[labeled == i] = 0

        # plt.imshow(final_mask,cmap='gray')
        # plt.title('mask')
        # plt.show()
        # 使用mask去掉落在外边的像素点
        masked_seg = cv2.bitwise_or(medsam_seg,medsam_seg, mask=final_mask)
        closed_seg = cv2.morphologyEx(masked_seg, cv2.MORPH_CLOSE, kernel)
        
        # result_image_name = os.path.join('./SAM_label/', batch["image_name"][0])
        result_image = Image.fromarray((closed_seg).astype(np.uint8))
        result_image.save(result_image_name)

        # plt.figure(figsize=(12,4))
        # plt.subplot(1,3,1)
        # plt.imshow(batch["pixel_values"][0,1], cmap='gray')
        # plt.title('original_image')
        # plt.axis('off')
        # plt.subplot(1,3,2)
        # plt.imshow(batch["ground_truth_mask"][0], cmap='copper')
        # plt.title('ground_truth_masks')
        # plt.axis('off')
        # plt.subplot(1,3,3)
        # plt.imshow(closed_seg, cmap='copper')
        # plt.title('after_canny')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        
        
        # if cnt>55:
        #     break
