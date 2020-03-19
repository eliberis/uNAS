import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import unittest
import numpy as np

from parameterized import parameterized_class
from tqdm import tqdm

from cnn import CnnSearchSpace
from mlp import MlpSearchSpace
from resource_models.models import peak_memory_usage, model_size, inference_latency


@parameterized_class(("search_space", "input_shape", "num_classes"), [
    (CnnSearchSpace(), (49, 40, 1), 12),
    (CnnSearchSpace(), (32, 32, 3), 10),
    (CnnSearchSpace(), (28, 28, 1), 10),
    (MlpSearchSpace(), (28, 28, 1), 10),
], class_name_func=lambda cls, idx, d: f"{cls.__name__}_{idx}_{d['search_space'].__class__.__name__}")
class SearchSpaceTests(unittest.TestCase):
    """Some fuzzing tests for the search space functions"""
    def setUp(self):
        np.random.seed(0)

    def _all_in_bounds(self, fv):
        for k, v in fv.items():
            self.assertTrue(self.search_space.schema[k].value_in_bounds(v),
                            msg=f"{v} not in bounds for {k}")

    def test_morphisms_produce_different_architectures(self):
        for _ in range(100):
            base = self.search_space.random_architecture()
            for m in self.search_space.produce_morphs(base):
                self.assertNotEqual(base.architecture, m.architecture)

    def test_resource_graph_assembly_succeeds(self):
        for i in range(5000):
            arch = self.search_space.random_architecture()
            self.search_space.to_resource_graph(arch, self.input_shape, self.num_classes)
        self.assertTrue(True)

    def test_resource_features_can_be_computed(self):
        for i in range(1000):
            arch = self.search_space.random_architecture()
            rg = self.search_space.to_resource_graph(arch, self.input_shape, self.num_classes)
            self.assertGreater(peak_memory_usage(rg), 0)
            self.assertGreater(model_size(rg), 0)
            self.assertGreater(inference_latency(rg), 0)

    def test_keras_model_assembly_succeeds(self):
        for i in tqdm(range(50)):
            arch = self.search_space.random_architecture()
            model = self.search_space.to_keras_model(arch, self.input_shape, self.num_classes)
            del model
        self.assertTrue(True)

    def test_resource_graph_assembly_for_morphs_succeeds(self):
        for i in range(1000):
            arch = self.search_space.random_architecture()
            for morph in self.search_space.produce_morphs(arch):
                self.search_space.to_resource_graph(morph, self.input_shape, self.num_classes)
        self.assertTrue(True)

    def test_keras_model_assembly_for_morphs_succeeds(self):
        for i in tqdm(range(5)):
            arch = self.search_space.random_architecture()
            for morph in self.search_space.produce_morphs(arch):
                model = self.search_space.to_keras_model(morph, self.input_shape, self.num_classes)
                del model
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
