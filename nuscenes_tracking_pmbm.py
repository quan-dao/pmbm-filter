import json
import argparse
import os
import numpy as np
from datetime import datetime

from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection


def track_one_scene(detection_file:str):
    np.random.seed(0)
    # initialize filter
    config = FilterConfig(state_dim=6, measurement_dim=3)
    density = get_gaussian_density_NuScenes_CV()
    pmbm_filter = PoissonMultiBernoulliMixture(config, density)

    # get measurements
    with open(detection_file, 'r') as f:
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
                                            for measurement, obj_type in zip(all_measurements[time_step], all_classes[time_step])]
    all_estimation = {}
    num_frames = len(all_object_detections.keys())
    for i_frame in range(num_frames):
        measurements = all_object_detections[str(i_frame)]
        # for meas in measurements:
        #     print(meas)
        # break

        print('Time step {}'.format(i_frame))
        all_estimation[i_frame] = pmbm_filter.run(measurements)

        print('After Update\n', pmbm_filter)
        print('\n-----------------------------\n')

    with open('./estimation-result/estimation-scene-0757-{}.json'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), 'w') as outfile:
        json.dump(all_estimation, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render tracking result in one scene of NuScenes.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detection_file', type=str, default='fixed-megvii-measurement-full-scene-0757.json',
                        help='Name of the detection file of the scene you want to track')
    args = parser.parse_args()
    pmbm_root = os.getcwd()
    detection_file = os.path.join(pmbm_root, 'scene-detection', args.detection_file)
    track_one_scene(detection_file)
