import logging
import ray
import os

import numpy as np
import tensorflow as tf
from functools import reduce, lru_cache


from dataset import Dataset


def copy_weight(source: tf.Tensor, destination: tf.Tensor):
    """
    Copies values from `src` to `dst`, making adjustments for each dimension, where necessary.
    :param source: Source tensor.
    :param destination: Destination tensor.
    """
    if source.shape == destination.shape:
        destination.assign(source)
        return

    dst = np.zeros(destination.shape)
    if len(source.shape) == 4:
        # Copying a conv kernel, which could be either from
        # Conv2D: with shape (K, K, C_in, C_out), or
        # DWConv2D: with shape (K, K, C, 1)
        sk_h, sk_w, sc_in, sc_out = source.shape
        dk_h, dk_w, dc_in, dc_out = dst.shape

        assert sk_h == sk_w and sk_h % 2 == 1
        assert dk_h == dk_w and dk_h % 2 == 1

        # If kernel size changes: copy `src` into the middle of `dst` or vice versa
        # For example:
        #
        #    5 x 5         3 x 3         3 x 3        5 x 5
        #  A B C D E                                0 0 0 0 0
        #  F G H I J       G H I         A B C      0 A B C 0
        #  K L M N O   =>  L M N    OR   D E F  =>  0 D E F 0
        #  P Q R S T       Q R S         G H I      0 G H I 0
        #  U V W X Y                                0 0 0 0 0
        #
        # If channels change: copy the first channels of `src` into the `dst` or vice versa

        # Compute offsets for the kernel dimension for source (sko) and destination (dko) tensors
        if sk_h < dk_h:
            dko, sko = (dk_h - sk_h) // 2, 0
        else:
            sko, dko = (sk_h - dk_h) // 2, 0

        # Compute how many channels (both in and out) will be transferred
        c_in = min(sc_in, dc_in)
        c_out = min(sc_out, dc_out)

        dst[dko:dk_h-dko, dko:dk_w-dko, :c_in, :c_out] = source[sko:sk_h-sko, sko:sk_w-sko, :c_in, :c_out]
    elif len(source.shape) == 2:
        # Copying a fully-connected (Dense) layer kernel; shape: (U_in, U_out)
        su_in, su_out = source.shape
        du_in, du_out = dst.shape

        u_in = min(su_in, du_in)
        u_out = min(su_out, du_out)
        dst[:u_in, :u_out] = source[:u_in, :u_out]
    else:
        # Copying a bias tensor
        assert len(source.shape) == 1
        c = min(source.shape[0], dst.shape[0])
        dst[:c] = source[:c]

    destination.assign(dst)


def quantised_accuracy(model: tf.keras.Model, dataset: Dataset,
                       batch_size: int, num_representative_batches=5,
                       num_eval_workers=4, output_file=None):
    """ Converts a Keras model into a quantised TF Lite model and reports its accuracy on the test
        set. Due to the slowness of the TF Lite interpreter, there's an option to parallelise
        evaluation on the test set using Ray.
        :param model A Keras model to quantise
        :param dataset Dataset that the model was trained with. Several batches from the
        validation subset will be used to calibrate the quantisation and the quantised model will be
        evaluated on the test subset.
        :param batch_size The batch size to use for evaluation. If this doesn't divide the test
        set evenly, the remainder will be dropped.
        :param num_representative_batches Number of validation subset batches to use for calibration
        :param num_eval_workers How many workers to use for evaluation (no parallelisation if = 1)
        :param output_file: Save the quantised TF lite model to this location (not written if None)
    """
    log = logging.getLogger("Quantiser")
    log.info("Computing quantised test accuracy...")

    # Add a batch dimension for evaluation speed
    if output_file is not None and batch_size != 1:
        print("Model output is requested, so the batch_size will be set to 1.")
        num_representative_batches = num_representative_batches * batch_size
        batch_size = 1

    def representative_dataset_gen():
        data = dataset.validation_dataset().batch(batch_size, drop_remainder=True)
        if num_representative_batches:
            data = data.take(num_representative_batches)
        for sample, _ in data:
            yield [sample]

    model.inputs[0].set_shape((batch_size, ) + dataset.input_shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    model_bytes = converter.convert()

    if output_file is not None:
        with open(output_file, "wb") as f:
            f.write(model_bytes)

    def evaluate(model_bytes, dataset, worker_id=0):
        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        accuracy = tf.keras.metrics.BinaryAccuracy() if dataset.num_classes == 2 else \
            tf.keras.metrics.SparseCategoricalAccuracy()
        test_data = dataset.test_dataset().batch(batch_size, drop_remainder=True)

        def filter_for_this_worker(i, x):
            return i % num_eval_workers == worker_id

        def discard_first(i, x):
            return x

        test_data = test_data.enumerate()\
            .filter(filter_for_this_worker) \
            .map(discard_first) \
            .as_numpy_iterator()

        for x, y_true in test_data:
            interpreter.set_tensor(input_index, x)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_index)
            accuracy.update_state(y_true, y_pred)
        return accuracy.total.numpy(), accuracy.count.numpy()

    if num_eval_workers > 1:
        ray.init(ignore_reinit_error=True)
        evaluate = ray.remote(num_cpus=1, num_return_vals=2)(evaluate)
        bytes_handle = ray.put(model_bytes)
        dataset_handle = ray.put(dataset)
        tasks = [evaluate.remote(bytes_handle, dataset_handle, i) for i in range(num_eval_workers)]
        total, count = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]), [ray.get(t) for t in tasks])
    else:
        assert num_eval_workers == 1
        total, count = evaluate(model_bytes, dataset)
    return total / count


class Scheduler:
    def __init__(self, gpu_trainers):
        self.workers_and_tasks = [(w, None) for w in gpu_trainers]

    def has_a_free_worker(self):
        return any(t is None for w, t in self.workers_and_tasks)

    def pending_tasks(self):
        return sum(t is not None for w, t in self.workers_and_tasks)

    def submit(self, *args, **kwargs):
        worker_idx = next(i for i, (w, t) in enumerate(self.workers_and_tasks) if t is None)
        assert worker_idx is not None

        worker = self.workers_and_tasks[worker_idx][0]
        self.workers_and_tasks[worker_idx] = (worker, worker.evaluate.remote(*args, **kwargs))

    def await_any(self):
        completed, pending = ray.wait([t for w, t in self.workers_and_tasks if t is not None],
                                      num_returns=1)
        assert len(completed) == 1
        task = completed[0]

        # Find out which worker completed and reset its task
        for i, (w, t) in enumerate(self.workers_and_tasks):
            if task == t:
                self.workers_and_tasks[i] = (w, None)
                break

        return ray.get(task)


def debug_mode():
    return bool(os.environ.get("UNAS_DEBUG"))


@lru_cache(maxsize=None)
def num_gpus():
    return len(tf.config.experimental.list_physical_devices("GPU"))
