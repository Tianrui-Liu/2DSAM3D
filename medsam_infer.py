from lungmask import LMInferer
import SimpleITK as sitk
from IPython.display import clear_output
import os
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create the lungmask inferer
inferer = LMInferer(tqdm_disable=True)

# Define the input and output directories
input_dir = "./dataset_512/test_2d_images_512/"
output_dir = "./dataset_512/infer_lungmask"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of test file paths
test_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.nii.gz')]

# Perform inference for each test file
for test_file in tqdm(test_files):
    
    # Read the NIfTI file using SimpleITK
    input_image = sitk.ReadImage(test_file)
    input_array = sitk.GetArrayFromImage(input_image)

    if len(input_array.shape)==2:
        volume_3d=input_array[np.newaxis,:,:]
        # Convert the 3D volume back to a SimpleITK image
        volume_3d_image = sitk.GetImageFromArray(volume_3d)
        # Now you can apply the lungmask package to the 3D image
        segmentation = inferer.apply(volume_3d_image)

    segmentation = segmentation.transpose((2, 1, 0)) 
    corrected_segmentation = np.fliplr(segmentation)
    
    # white_mask = corrected_segmentation > 0
    # corrected_segmentation[white_mask] = 255

    normalized_volume = ((volume_3d - volume_3d.min()) / (volume_3d.max() - volume_3d.min()) * 255).astype(np.uint8)
    edges_original = cv2.Canny(normalized_volume, 50, 200)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(edges_original, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积从大到小排序轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 第一个元素是最大的轮廓，第二个元素就是第二大的轮廓
    second_largest_contour = contours[1]

    # max_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(edges_original)
    # 在mask上画出最大的轮廓
    cv2.drawContours(mask, [second_largest_contour],-1, (255), thickness=cv2.FILLED)
    # 创建一个比输入图像大2的掩码，用于floodFill函数
    h, w = mask.shape[:2]
    mask_floodfill = np.zeros((h+2, w+2), np.uint8)
    # floodFill函数会改变输入图像，所以我们使用它的副本
    mask_floodfill_copy = mask.copy()
    # 找到一个种子点
    seed_point = (w//2, h//2)
    # 执行floodFill函数，将与种子点连通的区域填充为白色
    cv2.floodFill(mask_floodfill_copy, mask_floodfill, seed_point, 255)
    final_mask = mask | mask_floodfill_copy
    final_mask = cv2.resize(final_mask, (256, 256))

    # Ensure both images have the same size
    final_mask = cv2.resize(final_mask, (corrected_segmentation.shape[1], corrected_segmentation.shape[0]))

    # Ensure both images have the same data type (8-bit unsigned integer)
    final_mask = final_mask.astype(np.uint8)
    masked_seg = cv2.bitwise_and(corrected_segmentation,corrected_segmentation, mask=final_mask)
    closed_seg = cv2.morphologyEx(masked_seg, cv2.MORPH_CLOSE, kernel)
    # Save the result
    # print(test_file)
    # break
    png_filename = os.path.basename(test_file).rsplit('.', 2)[0] + '.png'
    result_image_path = os.path.join(output_dir, png_filename)
    plt.imshow(closed_seg, cmap='gray')
    plt.axis('off')
    plt.savefig(result_image_path)
    plt.close()
