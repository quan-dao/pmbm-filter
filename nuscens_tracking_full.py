import json
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox

from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection

np.random.seed(0)

NUSCENES_TRACKING_NAMES = [
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]


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
    # test pmbm tracking in val split of NuScenes
    detection_file = '/home/mqdao/Downloads/nuScene/detection-megvii/megvii_val.json'
    data_root = '/home/mqdao/Downloads/nuScene/v1.0-trainval'
    version = 'v1.0-trainval'

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # load detection
    with open(detection_file) as f:
        data = json.load(f)
    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']
    print('meta: ', meta)
    print("Loaded results from {}. Found detections for {} samples.".format(
        detection_file,
        len(all_results.sample_tokens)))
    # to filter detection
    all_score_theshold = [0.35, 0.3, 0.25, 0.2]

    # init tracking results
    tracking_results = {}

    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue

        # initialize filter
        config = FilterConfig(state_dim=6, measurement_dim=3)
        density = get_gaussian_density_NuScenes_CV()
        pmbm_filter = PoissonMultiBernoulliMixture(config, density)

        current_sample_token = nusc.get('scene', scene_token)['first_sample_token']
        current_time_step = 0

        while current_sample_token != '':
            # filter detections with low detection score
            sample_record = nusc.get('sample', current_sample_token)
            gt_num_objects = len(sample_record['anns'])
            filtered_detections = []
            i_threshold = 0
            while len(filtered_detections) < gt_num_objects and i_threshold < len(all_score_theshold):
                filtered_detections = [detection for detection in all_results.boxes[current_sample_token]
                                       if detection.detection_score >= all_score_theshold[i_threshold]]
                i_threshold += 1

            # create measurement for pmbm filter
            measurements = []
            for detection in filtered_detections:
                # get obj_type
                if detection.detection_name not in NUSCENES_TRACKING_NAMES:
                    continue
                obj_type = detection.detection_name
                # get object pose
                x, y, z = detection.translation
                quaternion = Quaternion(detection.rotation)
                yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
                # get object size
                size = list(detection.size)
                # get detection score
                score = detection.detection_score
                # create object detection
                obj_det = ObjectDetection(z=np.array([x, y, yaw]).reshape(3, 1),
                                          size=size,
                                          obj_type=obj_type,
                                          height=z,
                                          score=score,
                                          empty_constructor=False)
                measurements.append(obj_det)

            # print('Time {} - Number of measurements: {}'.format(current_time_step, len(measurements)))

            # initialize tracking results for this sample
            tracking_results[current_sample_token] = []

            # invoke filter and extract estimation
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
            current_sample_token = sample_record['next']
            current_time_step += 1

        processed_scene_tokens.add(scene_token)

    # save tracking result
    output_data = {'meta': meta, 'results': tracking_results}
    with open('./estimation-result/all-results-validataion-set-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), 'w') as outfile:
        json.dump(output_data, outfile)


if __name__ == '__main__':
    main()
