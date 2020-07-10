import json
import numpy as np
from datetime import datetime
from pyquaternion import Quaternion
from typing import List, Dict
from tqdm import tqdm

from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection

from nuscenes.nuscenes import NuScenes


def format_result(sample_token: str,
                  translation: List[float],
                  size: List[float],
                  yaw: float,
                  velocity: List[float],
                  tracking_id: int,
                  tracking_name: str,
                  tracking_score: float) -> Dict:
    """
    Format tracking result for 1 single target as following
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                           Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                           We average over frame level scores to compute the track level score.
                                           The score is used to determine positive and negative tracks via thresholding.
    }
    """
    sample_result = {}
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation
    sample_result['size'] = size
    sample_result['rotation'] = Quaternion(angle=yaw, axis=[0, 0, 1]).elements.tolist()
    sample_result['velocity'] = velocity
    sample_result['tracking_id'] = tracking_id
    sample_result['tracking_name'] = tracking_name
    sample_result['tracking_score'] = tracking_score
    return sample_result


def main():
    # load NuScenes
    data_root = '/home/mqdao/Downloads/nuScene/v1.0-trainval'
    version = 'v1.0-trainval'
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # load scene token
    with open('val_scene_tokens.json', 'r') as infile:
        val_scene_tokens = json.load(infile)

    # init tracking results for the whole val set
    tracking_results = {}

    for _, scene_token in tqdm(val_scene_tokens.items()):
        # get measurements
        with open('./megvii-per-scene-detection/val/{}.json'.format(scene_token), 'r') as f:
            data = json.load(f)
        all_measurements = data['all_measurements']
        all_classes = data['all_classes']
        all_object_detections = {}
        for time_step in all_classes.keys():
            all_object_detections[time_step] = [ObjectDetection(z=np.array(measurement[:3]).reshape(3, 1),
                                                                obj_type=obj_type,
                                                                size=measurement[3:6],
                                                                height=measurement[6],
                                                                score=measurement[7],
                                                                empty_constructor=False)
                                                for measurement, obj_type in
                                                zip(all_measurements[time_step], all_classes[time_step])]
        # initialize filter
        config = FilterConfig(state_dim=6, measurement_dim=3)
        density = get_gaussian_density_NuScenes_CV()
        pmbm_filter = PoissonMultiBernoulliMixture(config, density)

        current_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_time_step = 0

        while current_sample_token != '':
            # initialize tracking results for this sample
            tracking_results[current_sample_token] = []
            # invoke filter and extract estimation
            measurements = all_object_detections[str(current_time_step)]
            if len(measurements) > 0:
                estimation = pmbm_filter.run(measurements)
                # log estimation
                for target_id, target_est in estimation.items():
                    sample_result = format_result(current_sample_token,
                                                  target_est['translation'] + [target_est['height']],
                                                  target_est['size'],
                                                  target_est['orientation'],
                                                  target_est['velocity'],
                                                  target_id,
                                                  target_est['class'],
                                                  target_est['score'])
                    tracking_results[current_sample_token].append(sample_result)
            # move on
            current_sample_token = nusc.get('sample', current_sample_token)['next']
            current_time_step += 1

    # save tracking result
    meta = {'use_camera': False, 'use_lidar': True, 'use_radar': False, 'use_map': False, 'use_external': False}
    output_data = {'meta': meta, 'results': tracking_results}
    with open('./estimation-result/all-results-validataion-set-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), 'w') as outfile:
        json.dump(output_data, outfile)


if __name__ == '__main__':
    main()