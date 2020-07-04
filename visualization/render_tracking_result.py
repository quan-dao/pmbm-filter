import json
from pyquaternion import Quaternion
import cv2
import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

from box_for_track import Box4Track


def main():
    # Load tracking data
    with open('./../estimation-result/estimation-scene-0757-20200704-165858.json') as infile:
        all_tracking_result = json.load(infile)

    num_unique_colors = 200
    all_color_indicies = np.linspace(0, 1.0, num=num_unique_colors)  # allow up to 200 unique colors

    # load NuScenese styff
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/mqdao/Downloads/nuScene/v1.0-mini', verbose=False)
    my_scene_token = nusc.field2token('scene', 'name', 'scene-0757')[0]
    my_scene = nusc.get('scene', my_scene_token)

    current_time_step = 0
    current_sample_token = my_scene['first_sample_token']
    while True:
        # get necessary record
        sample_record = nusc.get('sample', current_sample_token)
        camera_token = sample_record['data']['CAM_FRONT']
        sd_record = nusc.get('sample_data', camera_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # get camera information
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
        impath = nusc.get_sample_data_path(camera_token)
        im = cv2.imread(impath)

        # get tracking result
        current_tracks = all_tracking_result[str(current_time_step)]
        for target_id, target in current_tracks.items():
            box = Box4Track(center=target['translation'] + [target['height']],
                            orientation=Quaternion(axis=[0, 0, 1], angle=target['orientation']),
                            size=target['size'],
                            name=target['class'],
                            label=int(target_id)
                            )

            box.to_camera_frame(pose_record, cs_record)

            # render box on image
            if not box_in_image(box, cam_intrinsic, imsize, BoxVisibility.ANY):
                # print('Box {} not in image'.format(box.name))
                continue
            # get color
            c = np.array(plt.cm.Spectral(box.label % num_unique_colors))
            c = np.round(c * 255)
            box.render_track(im, view=cam_intrinsic, normalize=True, color=(c[0], c[1], c[2]))

        # move on
        current_time_step += 1
        current_sample_token = sample_record['next']
        if current_sample_token == '':
            break

        cv2.imshow('CAM_FRONT', im)
        key = cv2.waitKey(500)  # wait 100ms
        if key == 32:  # if space is pressed, pause.
            key = cv2.waitKey()
        if key == 27:  # if ESC is pressed, exit.
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
