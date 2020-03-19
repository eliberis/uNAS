# μNAS

μNAS (micro-NAS or mu-NAS) is a neural architecture search system that specialises in finding
 ultra-small models suitable for deploying on microcontrollers: think < 64 KB memory and storage
 requirement. μNAS achieves this by explicitly targeting three primary resource bottlenecks:
 model size, latency and peak memory usage.

For a full description of methodology and experimental results, please see the accompanying paper
 _"μNAS: Constrained Neural Architecture Search for Microcontrollers"_ (_URL TBA_). 
 
 
## Usage 
 
### Setup
 
μNAS uses Python 3.7+ with the environment described by `Pipfile`: to create an
 environment with all correct packages preinstalled simply run `pipenv install` in the cloned
 repository. 

### To run

The search is configured using Python configuration files (see `configs` for examples and 
`config.py` for configuration file schema), which specify the search algorithm, how candidate models
are going to be trained (incl. any pruning configuration) and resource bounds. μNAS can be invoked
using `driver.py` which immediately delegates to the configured search algorithm. 

For example, to search for MNIST models with Aging Evolution and structured pruning, run the
 following:

`pipenv run python driver.py configs/cnn_mnist_struct_pru.py --name "example_mnist"`

## Navigating the code

- `cnn`/`mlp`: contains a search space description for convolutional neural networks / multilayer
 perceptrons, together with all allowed morphisms (changes) to a candidate architecture.

- `configs`: example search configurations,

- `dataset`: loaders for various datasets, conforming to the interface in `dataset/dataset
.py`

- `dragonfly_adapters`: (Bayesian optimisation only) extra code to interoperate with 
[Dragonfly](https://github.com/dragonfly/dragonfly). We found that we had to rely on internal
 implementation of the framework for it to correctly use our customised kernel, search space and
 a genetic algorithm optimiser for acq. functions, thus the module contains a fair amount of
  monkey-patches.
  
- `resource_models`: an independent library that allows representing and computing resource usage
 of arbitrary computation graphs.
 
- `search_algorithms`: implements aging evolution and Bayesian optimisation search algorithms;
 each search algorithm is also responsible for scheduling model training and correctly
 serialising & restoring the search state. Both use `ray` under the hood to parallelise the search.
 
- `teachers`: a collection of teacher models for distillation.

- `test`: automated sanity tests for search space implementations.

- `model_trainer.py`: code for training candidate models.

- `pruning.py`: implements [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB)
 as a Keras callback, used during training.

- `generate_tflite_models.py`: generates random small models for latency benchmarking on a
 microcontroller.
 
- `search_state_processor.py`: loads and visualises μNAS search state files. 
 
- `architecture.py`/`config.py`/`search_space.py`/`schema_types.py` base classes for candidate
 architectures, search configuration and free variables of the search space.


## Notes on deploying found models

In the interest of storage, μNAS does not save final weights of discovered models (though
 it can be modified to do so): μNAS uses aging evolution and does not share trained weights
 across candidate models, which encourages finding models that can be trained to good accuracy
 from scratch. You can easily instantiate a Keras model from a found architecture (see
  API in `architecture.py`).
 
 μNAS assumes a runtime where each operator is executed one
 at a time and in full, such as ["TensorFlow Lite Micro"](https://www.tensorflow.org/lite/microcontrollers). 
 You can quantise and convert Keras models to the TFLite format using helper functions in `utils.py`.
  Note that:
  
 - μNAS only calculates resource usage of a model and does not take particular framework overheads
  into account.
  
 - μNAS assumes that one of the input buffers to an `Add` operator can be reused as an output buffer
 if it is not used elsewhere (to minimise peak memory usage); this optimisation is not available
  in TF Lite Micro at the time of writing.
 
 - The operator execution order that gives the smallest peak memory usage is not recorded in the
  model: use [`tflite-tools`](https://github.com/eliberis/tflite-tools) to optimise your tflite
   model prior to deploying.


