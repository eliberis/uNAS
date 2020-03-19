import numpy as np
import tensorflow as tf

from architecture import Architecture
from cnn import CnnSearchSpace
from resource_models.models import model_size, peak_memory_usage


def main():
    np.random.seed(0)

    num_models = 1000
    output_dir = "/tmp/tflite"

    ss = CnnSearchSpace()

    input_shape = (64, 64, 3)
    num_classes = 10
    ms_req, pmu_req = 250_000, 250_000

    def get_resource_requirements(arch: Architecture):
        rg = ss.to_resource_graph(arch, input_shape, num_classes)
        return model_size(rg), peak_memory_usage(rg, exclude_inputs=False)

    def evolve_until_within_req(arch):
        keep_prob = 0.25
        ms, pmu = get_resource_requirements(arch)

        while ms > ms_req or pmu > pmu_req:
            morph = np.random.choice(ss.produce_morphs(arch))
            new_ms, new_pmu = get_resource_requirements(morph)
            if new_ms < ms or new_pmu < pmu or np.random.random_sample() < keep_prob:
                ms, pmu = new_ms, new_pmu
                arch = morph

        return arch

    def convert_to_tflite(arch: Architecture, output_file):
        model = ss.to_keras_model(arch, input_shape, num_classes)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = \
            lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        model_bytes = converter.convert()

        if output_file is not None:
            with open(output_file, "wb") as f:
                f.write(model_bytes)

    for i in range(num_models):
        print(f"Generating #{i + 1}...")
        arch = evolve_until_within_req(ss.random_architecture())
        convert_to_tflite(arch, output_file=f"{output_dir}/m{i:05d}.tflite")


if __name__ == '__main__':
    main()
