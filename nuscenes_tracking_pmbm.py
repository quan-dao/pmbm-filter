import json
import numpy as np

from filter_config import FilterConfig, get_gaussian_density_NuScenes_CV
from pmbm import PoissonMultiBernoulliMixture
from object_detection import ObjectDetection


def main():
    np.random.seed(0)
    # initialize filter
    config = FilterConfig(state_dim=6, measurement_dim=3)
    density = get_gaussian_density_NuScenes_CV()
    pmbm_filter = PoissonMultiBernoulliMixture(config, density)

    # get measurements
    with open('meausrement-scene-0757.json', 'r') as f:
        data = json.load(f)
    all_measurements = data['all_measurements']
    all_classes = data['all_classes']
    all_object_detections = {}
    for time_step in all_classes.keys():
        all_object_detections[time_step] = [ObjectDetection(z=np.array(measurement).reshape(3, 1),
                                                            obj_type=obj_type,
                                                            empty_constructor=False)
                                            for measurement, obj_type in zip(all_measurements[time_step], all_classes[time_step])]

    num_frames = len(all_object_detections.keys())
    for i_frame in range(num_frames):
        measurements = all_object_detections[str(i_frame)]
        # for meas in measurements:
        #     print(meas)
        # break
        print('Time step {}'.format(i_frame))
        # print('Before Prediction \n', pmbm_filter)

        pmbm_filter.run(measurements)

        print('After Update\n', pmbm_filter)
        print('\n-----------------------------\n')

        # early stop
        if i_frame > 15:
            break


if __name__ == '__main__':
    main()
