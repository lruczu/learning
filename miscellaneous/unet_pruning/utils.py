from collections import defaultdict
import os
import re
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tqdm import tqdm

from src.segmentation_lite.data_utils import load_all_pairs
from src.segmentation_lite.preprocessor_config import PreprocessorConfig
from src.utils.other import unzip_to_arrays


def soft_threshold(x: tf.Tensor, s: tf.Tensor) -> tf.Tensor:
    return tf.sign(x) * tf.nn.relu(tf.abs(x) - tf.math.sigmoid(s))


def zero_count(x: tf.Tensor) -> int:
    return tf.reduce_sum(
        tf.cast(x == 0, tf.float32)
    ).numpy()


def zero_proportion(x: tf.Tensor) -> float:
    return tf.reduce_mean(
        tf.cast(x == 0, tf.float32)
    ).numpy()


def get_name_of_layer_from_tensor(x: tf.Tensor):
    """
    It only works for tensors from tensorflow Model.

    Parameters
    ----------
    x: input of some layer

    Returns
    -------
        name of layer from which input 'x' comes from
    """
    match = re.search(r'created by layer \'(.+)\'\)\>', x.__repr__())
    if match is None:
        raise ValueError('Tensor does not come from tensorflow model')

    return match.group(1)


def copy_layer(
    layer: tf.keras.layers.Layer,
    layer_name_suffix: str = '',
    overwrite_params: Optional[Dict[str, Any]] = None,
) -> tf.keras.layers.Layer:
    layer_attrs = layer.get_config()
    if overwrite_params is not None:
        layer_attrs.update(overwrite_params)
    layer_attrs.update({
        'name': layer_attrs['name'] + layer_name_suffix
    })
    return layer.__class__.from_config(layer_attrs)


def get_magnitudes(kernel: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_sum(kernel * kernel, axis=[0, 1, 2]))


def get_absolute(kernel: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.abs(kernel), axis=[0, 1, 2])


def get_layer_mapping(model):
    """
    Returns:
        (layer name: str -> next layers: List[str])
    """
    mapping = defaultdict(list)

    for layer in model.layers[1:]:
        inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        for i in inputs:
            previous_layer_name = get_name_of_layer_from_tensor(i)
            mapping[previous_layer_name].append(layer.name)

    return dict(mapping)


def get_n_params(kernel: tf.Tensor) -> int:
    return tf.size(kernel).numpy()


def get_n_zeros(kernel: tf.Tensor, s: tf.Tensor) -> int:
    return zero_count(soft_threshold(kernel, s))


def compute_sparsity(sparse_model):
    n_params = 0
    n_zeros = 0
    s_values = []
    for layer in sparse_model.layers:
        for w in layer.get_weights():
            n_params += tf.size(w).numpy()

        if type(layer).__name__ == 'SparseConv2D':
            n_zeros += get_n_zeros(layer.kernel, layer.s)
            s_values.append(layer.s.numpy())

        elif type(layer).__name__ == 'StructuredSparseConv2D':
            n_zeros += zero_count(soft_threshold(layer.kernel, layer.s))
            s_values.append(layer.s.numpy())

    return n_zeros / n_params, np.min(s_values), np.max(s_values)


def save_quantized_model(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_output_type = tf.float32
    converter.inference_input_type = tf.float32
    converter.experimental_new_converter = False

    tflite_model = converter.convert()

    output_path = os.path.join(path)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def get_quantization_generator(config: dict, n_samples: int = 50):
    """Used for quantization to estimate distribution of activation values."""

    samples_from_train = load_all_pairs(set_type='train', limit=n_samples)
    preprocessor_config = PreprocessorConfig(
        input_size=config['input_size'],
        aug_size=config['aug_size'],
        full_size=config['full_size'],
    )
    x, _ = unzip_to_arrays(map(lambda pair: preprocessor_config.preprocess_test(*pair),
                               samples_from_train))
    dataset = tf.data.Dataset.from_tensor_slices(x)

    def generator():
        for image in dataset:
            yield [tf.expand_dims(image, axis=0)]
    return generator


def get_iou_for_tflite(model_path: str, x: np.ndarray, y: np.ndarray, n_threads: int = 4):
    n = x.shape[0]

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=n_threads)
    iou = sm.metrics.IOUScore(per_image=True)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scores = []
    for i in tqdm(range(n)):  # it is very slow so batching does not make sense
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], x[[i]])
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[:, :, :, 0]

        score = iou(y[[i]], output).numpy()
        scores.append(score)

    return np.mean(scores)


def model_size(model) -> int:
    size = 0
    for layer in model.layers:
        for w in layer.get_weights():
            size += tf.size(w).numpy()
    return size


def n_filters_per_layer(model) -> Dict[str, int]:
    d = {}
    for layer in model.layers:
        if type(layer).__name__ == 'Conv2D':
            d[layer.name] = layer.filters
    return d
