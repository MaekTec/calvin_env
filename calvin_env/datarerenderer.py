#!/usr/bin/python3
import matplotlib.pyplot as plt
from calvin_env.envs.play_table_env import get_env
from calvin_env.camera.tactile_sensor import TactileSensor
from calvin_env.utils.occlusion_boundary import render_occlusion_boundary
from calvin_env.utils.surface_normals import render_normals
from pathlib import Path
import matplotlib.pyplot as plt
import hydra
import numpy as np
import omegaconf
import logging
import glob
import os

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_rerendering")
def main(cfg):
    log.info("pyBullet Data Rerenderer")
    log.info("Determining episodes to rerender")

    # Determine episodes to rerender
    recording_dir = (Path(hydra.utils.get_original_cwd()) / cfg.load_dir).absolute()
    rendered_episodes = map(lambda f: f.split("/")[-1], glob.glob(os.path.join(recording_dir, "*.npz")))
    prev_rerendered_episodes = map(lambda f: f.split("/")[-1], glob.glob(os.path.join(os.getcwd(), "*.npz")))
    episodes = list(set(rendered_episodes).difference(set(prev_rerendered_episodes)))
    episodes.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    episodes = list(map(lambda f: os.path.join(recording_dir, f), episodes))

    log.info(f"Found {len(episodes)} episodes to rerender")

    if cfg.processes != 1 and cfg.show_gui:
        log.warning("Multiprocess rendering requires headless mode, setting cfg.show_gui = False")
        cfg.show_gui = False

    env = get_env(recording_dir, show_gui=cfg.show_gui)

    log.info("Initialization done!")

    for episode in episodes[0:10]:
        log.info(f"Rerender {episode}")
        data = np.load(episode)
        env.reset(robot_obs=data["robot_obs"], scene_obs=data["scene_obs"])

        #for k in data.files:
        #    print(k)

        normals = []
        occlusion_boundaries = []
        cam_names = []
        for cam in env.cameras:
            if not isinstance(cam, TactileSensor):
                rgb_img, depth_img, segmentation_mask = cam.render(scale_factor=cfg.scale_factor, process_segmentation_mask=True)
                n = render_normals(cam, cfg.scale_factor, depth_img)
                print(n.shape)
                #n = np.transpose(n, axes=[1, 2, 0])
                n -= np.min(n)
                n = n / np.max(n)
                plt.imshow(n)
                plt.show()
                normals.append(n)
                oc = render_occlusion_boundary(cam, depth_img, segmentation_mask)
                plt.imshow(oc)
                plt.show()
                occlusion_boundaries.append(oc)
                cam_names.append(cam.name)
        
        normals_entries = {f"normals_{cam_name}": normals[i] for i, cam_name in enumerate(cam_names)}
        occlusion_boundaries_entries = {f"occlusion_boundary_{cam_name}": occlusion_boundaries[i] for i, cam_name in enumerate(cam_names)}
        file_name = episode.split("/")[-1]
        np.savez_compressed(
            file_name,
            **data,
            **normals_entries,
            **occlusion_boundaries_entries,
        )

        data.close()

    print("Done")

if __name__ == "__main__":
    main()

