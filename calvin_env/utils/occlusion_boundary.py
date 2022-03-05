import numpy as np
import cv2
import logging

# A logger for this file
logger = logging.getLogger(__name__)

# OcclusionBoundary
# Adapted from: https://github.com/Shreeyak/cleargrasp/blob/master/z-ignore-scripts-helper/data_processing_script.py
    
def gradient(img, kernel_size, threshold):
    # Apply Sobel Filter
    img = np.expand_dims(img, -1).astype(np.float32)
    #img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)

    sobelx = np.abs(sobelx)
    sobely = np.abs(sobely)

    # Create Boolean Mask
    sobelx_binary = np.full(sobelx.shape, False, dtype=bool)
    sobelx_binary[sobelx >= threshold] = True

    sobely_binary = np.full(sobely.shape, False, dtype=bool)
    sobely_binary[sobely >= threshold] = True

    sobel_binary = np.logical_or(sobelx_binary, sobely_binary)

    sobel_result = np.zeros_like(img, dtype=np.uint8)
    sobel_result[sobel_binary] = 1
    sobel_result = sobel_result[:, :, 0]

    return sobel_result # (height, width, 1)

def render_occlusion_boundary(camera, depth_img, segmentation_mask):
    # Gradient on depth
    depth_edges = gradient(depth_img, 5, 0.6)
    kernel = np.ones((5, 5), np.uint8)
    depth_edges = cv2.dilate(depth_edges, kernel, iterations=1)
    depth_edges_thik = cv2.dilate(depth_edges, kernel, iterations=1)

    # Gradient on segmentation mask
    mask_edges = gradient(segmentation_mask, 3, 1)

    # Calculate contact egdes
    contact_edges = mask_edges * (1-depth_edges_thik)
    kernel = np.ones((5, 5), np.uint8)
    contact_edges = cv2.dilate(contact_edges, kernel, iterations=1)

    # downsize 
    depth_edges = cv2.resize(depth_edges, (camera.width, camera.height), interpolation=cv2.INTER_CUBIC) > 0
    contact_edges = cv2.resize(contact_edges, (camera.width, camera.height), interpolation=cv2.INTER_CUBIC) > 0

    combined_outlines = np.zeros_like(depth_edges, dtype=np.uint8)
    combined_outlines[depth_edges] = 1
    combined_outlines[contact_edges] = 2
    return combined_outlines