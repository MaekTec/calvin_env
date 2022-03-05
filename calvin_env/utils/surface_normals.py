import logging
import numpy as np
import cv2

# A logger for this file
logger = logging.getLogger(__name__)

# Normals
# Adapted from https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_surface_normals.py

# A very simple and slow function to calculate the surface normals from 3D points from
# a reprojected depth map. A better method would be to fit a local plane to a set of 
# surrounding points with outlier rejection such as RANSAC.  Such as done here:
# http://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html
def normals_from_point_cloud(points, height, width):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 1
    lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}
    #surface_normals_s = np.zeros((height,width,3))
    surface_normals = np.zeros((height,width,3))
    
    # Unvectorized verison
    """
    for i in range(height):
        for j in range(width):
            min_diff = None
            point1 = points[i,j,:3]
            # We choose the normal calculated from the two points that are
            # closest to the anchor points.  This helps to prevent using large
            # depth disparities at surface borders in the normal calculation.
            for k in range(8):
                try:
                    point2 = points[i+lookups[k][0],j+lookups[k][1],:3]
                    point3 = points[i+lookups[(k+2)%8][0],j+lookups[(k+2)%8][1],:3]
                    diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                    if min_diff is None or diff < min_diff:
                        normal = NormalsRenderer.normalize(np.cross(point2-point1,point3-point1))
                        min_diff = diff
                except IndexError:
                    # Remark: Incorrect, because this doesn't happen on left/down boundary,
                    # points[-1] takes last index instead of throwing an IndexError
                    continues
            surface_normals_s[i,j,:3] = normal
    """
    
    # Vectorized version
    # Much faster, but can be wrong at the boundary in an unlikely edge case.
    # If this case appears, just crop the output by one pixel on each side.
    diff = np.zeros((height, width, 8))
    normal = np.zeros((height, width, 8, 3), dtype=np.float32)
    for k in range(8):
        points1 = points
        points2 = np.roll(points, (lookups[k][0], lookups[k][1]), axis=(0, 1))
        points3 = np.roll(points, (lookups[(k+2)%8][0], lookups[(k+2)%8][1]), axis=(0, 1))
        diff_points = np.linalg.norm(points2 - points1, axis=2) + np.linalg.norm(points3 - points1, axis=2) # (height, width)
        diff[:, :, k] = diff_points
        normal_points = -1 * np.cross(points2 - points1, points3 - points1, axis=2) # (height, width, 3)
        normal_points /= np.expand_dims(np.linalg.norm(normal_points, axis=2), 2) # (height, width, 3)
        normal[:, :, k] = normal_points
    a = np.expand_dims(np.argmin(diff, 2), 2) # (height, width, 1)
    a = np.stack((a, a, a), 3) # (height, width, 1, 3)
    surface_normals = np.take_along_axis(normal, a, 2) # (height, width, 1, 3)
    surface_normals = surface_normals[:, :, 0, :] # (height, width, 3)
    #print(np.allclose(surface_normals_s[2:-2, 2:-2, :], surface_normals[2:-2, 2:-2, :]))

    return surface_normals

def render_normals(camera, scale_factor, depth_img):
    fov = camera.fov
    width = camera.width
    height = camera.height
    points_in_camera = camera.distance_map_to_point_cloud(depth_img, fov, width*scale_factor, height*scale_factor)
    surface_normals = normals_from_point_cloud(points_in_camera, height*scale_factor, width*scale_factor) # (height, width, 3)
    surface_normals = cv2.resize(surface_normals, (width, height), interpolation=cv2.INTER_AREA)
    return surface_normals
