import numpy as np
import pybullet as p

from calvin_env.camera.camera import Camera


class StaticCamera(Camera):
    def __init__(
        self,
        fov,
        aspect,
        nearval,
        farval,
        width,
        height,
        look_at,
        look_from,
        up_vector,
        cid,
        name,
        robot_id=None,
        objects=None,
    ):
        """
        Initialize the camera
        Args:
            argument_group: initialize the camera and add needed arguments to argparse

        Returns:
            None
        """
        self.nearval = nearval
        self.farval = farval
        self.fov = fov
        self.aspect = aspect
        self.look_from = look_from
        self.look_at = look_at
        self.up_vector = up_vector
        self.width = width
        self.height = height
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self.nearval, farVal=self.farval
        )
        self.cid = cid
        self.name = name

    def set_position_from_gui(self):
        info = p.getDebugVisualizerCamera(physicsClientId=self.cid)
        look_at = np.array(info[-1])
        dist = info[-2]
        forward = np.array(info[5])
        look_from = look_at - dist * forward
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=look_from, cameraTargetPosition=look_at, cameraUpVector=self.up_vector
        )
        look_from = [float(x) for x in look_from]
        look_at = [float(x) for x in look_at]
        return look_from, look_at

    def render(self, scale_factor=1, process_segmentation_mask=False):
        image = p.getCameraImage(
            width=self.width * scale_factor,
            height=self.height * scale_factor,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            physicsClientId=self.cid,
        )
        if process_segmentation_mask:
            rgb_img, depth_img, segmentation_mask = self.process_rgbd(image, self.nearval, self.farval, process_segmentation_mask=True)
            return rgb_img, depth_img, segmentation_mask
        else:
            rgb_img, depth_img = self.process_rgbd(image, self.nearval, self.farval, process_segmentation_mask=False)
            return rgb_img, depth_img
