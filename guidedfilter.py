import cv2
import numpy as np

left_image = cv2.imread('/home/tharun/Data_extended/Baby1/view1.png', 0)
right_image = cv2.imread('/home/tharun/Data_extended/Baby1/view5.png', 0)
left_image = np.float32(left_image) / 255.0
right_image = np.float32(right_image) / 255.0

radius = 6
eps = 0.01
left_filtered = cv2.ximgproc.guidedFilter(left_image, left_image, radius, eps)
right_filtered = cv2.ximgproc.guidedFilter(right_image, right_image, radius, eps)
left_filtered = np.uint8(left_filtered * 255)
right_filtered = np.uint8(right_filtered * 255)

window_size = 5
min_disparity = 0
num_disparities = 128
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2
)
disparity = stereo.compute(left_filtered, right_filtered).astype(np.float32) / 16.0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_HH4        #Using gradient transform
)
disparity_gt=stereo.compute(left_filtered, right_filtered).astype(np.float32) / 16.0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY   #Using rank transform
)
disparity_rt=stereo.compute(left_filtered, right_filtered).astype(np.float32) / 16.0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_HH         #Using Hamming distance
)
disparity_hh=stereo.compute(left_filtered, right_filtered).astype(np.float32) / 16.0

# Post-processing
disparity = cv2.medianBlur(disparity, 5)  
disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))  
disparity = cv2.GaussianBlur(disparity, (5, 5), 0)  

disp_min = disparity.min()
disp_max = disparity.max()
disparity_normalized = ((disparity - disp_min) / (disp_max - disp_min)) * 255.0
disparity_processed = np.uint8(disparity_normalized)

disparity_gt = cv2.medianBlur(disparity_gt, 5)  
disparity_gt = cv2.morphologyEx(disparity_gt, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) 
disparity_gt = cv2.GaussianBlur(disparity_gt, (5, 5), 0) 

disp_min = disparity_gt.min()
disp_max = disparity_gt.max()
disparity_gt_normalized = ((disparity_gt - disp_min) / (disp_max - disp_min)) * 255.0
disparity_gt_processed = np.uint8(disparity_gt_normalized)

disparity_rt = cv2.medianBlur(disparity_rt, 5) 
disparity_rt = cv2.morphologyEx(disparity_rt, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) 
disparity_rt = cv2.GaussianBlur(disparity_rt, (5, 5), 0)  

disp_min = disparity_rt.min()
disp_max = disparity_rt.max()
disparity_rt_normalized = ((disparity_rt - disp_min) / (disp_max - disp_min)) * 255.0
disparity_rt_processed = np.uint8(disparity_rt_normalized)

disparity_hh = cv2.medianBlur(disparity_hh, 5)  
disparity_hh = cv2.morphologyEx(disparity_hh, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)) 
disparity_hh = cv2.GaussianBlur(disparity_hh, (5, 5), 0) 

disp_min = disparity_hh.min()
disp_max = disparity_hh.max()
disparity_hh_normalized = ((disparity_hh - disp_min) / (disp_max - disp_min)) * 255.0
disparity_hh_processed = np.uint8(disparity_hh_normalized)

cv2.imwrite('guidedimagefilter1.png', disparity_processed)
cv2.imwrite('guidedimagefilter2.png', disparity_rt_processed)
cv2.imwrite('guidedimagefilter3.png', disparity_gt_processed)
cv2.imwrite('guidedimagefilter4.png', disparity_hh_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()
