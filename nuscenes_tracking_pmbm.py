import json
import numpy as np

from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection


def main():
    # initialize filter
    config = FilterConfig(state_dim=6, measurement_dim=3)
    density = get_gaussian_density_NuScenes_CV()
    pmbm_filter = PoissonMultiBernoulliMixture(config, density)

    # get measurements
    with open('test_scene_measurement.json', 'r') as f:
        data = json.load(f)
    all_measurements = data['measurements']
    all_class = data['classes']

    num_frames = len(all_measurements.keys())
    for i_frame in range(num_frames):
        all_z_in_this_frame = [np.array(meas).reshape(-1, 1) for meas in all_measurements[str(i_frame)]]
        classes = all_class[str(i_frame)]
        measurements = [ObjectDetection(z, obj_type, empty_constructor=False)
                        for z, obj_type in zip(all_z_in_this_frame, classes)]
        # for meas in measurements:
        #     print(meas)

        print('Before Prediction \n', pmbm_filter)

        pmbm_filter.run(measurements)

        print('After Update\n', pmbm_filter)
        print('\n-----------------------------\n')
        # stop after 1 frame
        if i_frame > 3:
            break


if __name__ == '__main__':
    main()
