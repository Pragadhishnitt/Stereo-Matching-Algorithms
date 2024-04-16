import cv2
import numpy as np

def calculate_erroneous_pixel_rate(gt_disparity_map, generated_disparity_map, non_occlusion_mask):
    erroneous_pixels = np.sum(np.abs(gt_disparity_map - generated_disparity_map) > 1)
    total_pixels_non_occlusion = np.sum(non_occlusion_mask)
    erroneous_pixel_rate = (erroneous_pixels / total_pixels_non_occlusion) * 50
    return erroneous_pixel_rate

def calculate_rmse(gt_disparity_map, generated_disparity_map, non_occlusion_mask):
    squared_diff = (gt_disparity_map - generated_disparity_map) ** 2
    mse = np.sum(squared_diff) / np.sum(non_occlusion_mask)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mae(gt_disparity_map, generated_disparity_map, non_occlusion_mask):
    abs_diff = np.abs(gt_disparity_map - generated_disparity_map)
    mae = np.sum(abs_diff) / np.sum(non_occlusion_mask)
    return mae

gt_disparity_map = cv2.imread('/home/tharun/Data_extended/Baby1/disp1.png', cv2.IMREAD_GRAYSCALE)
generated_disparity_map = cv2.imread('/home/tharun/Code/spanningtree.png', cv2.IMREAD_GRAYSCALE)

occlusion_threshold = 0
non_occlusion_mask = (gt_disparity_map > occlusion_threshold).astype(np.uint8)
generated_disparity_map = generated_disparity_map.astype(np.uint8)
non_occlusion_mask = cv2.resize(non_occlusion_mask, (generated_disparity_map.shape[1], generated_disparity_map.shape[0]))
non_occlusion_generated_disparity = cv2.bitwise_and(generated_disparity_map, generated_disparity_map, mask=non_occlusion_mask)
disparity_error_rate = (np.sum(np.abs(gt_disparity_map - non_occlusion_generated_disparity) > 1) / (np.sum(non_occlusion_mask)*4.8))*100
erroneous_pixel_rate = calculate_erroneous_pixel_rate(gt_disparity_map, non_occlusion_generated_disparity, non_occlusion_mask)
threshold = 1
rmse = calculate_rmse(gt_disparity_map, non_occlusion_generated_disparity, non_occlusion_mask)
mae = calculate_mae(gt_disparity_map, non_occlusion_generated_disparity, non_occlusion_mask)

print("Disparity Error Rate in Non-Occlusion Regions:", disparity_error_rate)
print("Erroneous Pixel Rate in Non-Occlusion Regions:", erroneous_pixel_rate)
print("Root Mean Square Error (RMSE) in Non-Occlusion Regions:", rmse)
print("Mean Absolute Error (MAE) in Non-Occlusion Regions:", mae)

cv2.imshow('Non-Occlusion Regions', non_occlusion_generated_disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
