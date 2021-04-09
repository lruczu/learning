from typing import List

import tensorflow as tf

from src.pruning.utils import copy_layer


class ShrinkageManager:
    @staticmethod
    def shrink_block(
        model,
        conv2d: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
        flow,
        mapping,
    ):
        new_conv2d = ShrinkageManager.shrink_conv_layer(conv2d, indices_to_remove)
        flow = new_conv2d(flow)
        block_dict = {new_conv2d.name: new_conv2d}
        current_name = conv2d.name
        last_name = current_name.replace('expand', 'project')  # last layer affected by change

        while current_name != last_name:
            current_name = mapping[current_name][0]  # next layer name
            layer = model.get_layer(current_name)
            fn = ShrinkageManager.get_shrinkage_fn(current_name)
            if fn is None:
                layer_ = copy_layer(layer)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_
            else:
                layer_ = fn(layer, indices_to_remove, conv2d)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_

        return flow, block_dict

    @staticmethod
    def shrink_decoder(
        model,
        conv2d: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
        flow,
        mapping,
    ):
        new_conv2d = ShrinkageManager.shrink_conv_layer(conv2d, indices_to_remove)
        flow = new_conv2d(flow)
        block_dict = {new_conv2d.name: new_conv2d}
        current_name = conv2d.name
        last_name = current_name.replace('a_conv', 'b_conv')

        while current_name != last_name:
            current_name = mapping[current_name][0]  # next layer name
            layer = model.get_layer(current_name)
            if current_name == last_name:
                fn = ShrinkageManager.shrink_project_layer
            else:
                fn = ShrinkageManager.get_shrinkage_fn(current_name)

            if fn is None:
                layer_ = copy_layer(layer)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_
            else:
                layer_ = fn(layer, indices_to_remove, conv2d)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_

        return flow, block_dict

    @staticmethod
    def shrink_conv1(
        model,
        conv2d: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
        flow,
        mapping,
    ):
        assert conv2d.name == 'Conv_1'
        new_conv2d = ShrinkageManager.shrink_conv_layer(conv2d, indices_to_remove)
        flow = new_conv2d(flow)
        block_dict = {new_conv2d.name: new_conv2d}
        current_name = conv2d.name
        while current_name != 'decoder_stage0_upsampling':
            current_name = mapping[current_name][0]  # next layer name
            layer = model.get_layer(current_name)
            fn = ShrinkageManager.get_shrinkage_fn(current_name)
            if fn is None:
                layer_ = copy_layer(layer)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_
            else:
                layer_ = fn(layer, indices_to_remove, conv2d)
                flow = layer_(flow)
                block_dict[layer_.name] = layer_

        return flow, block_dict

    @staticmethod
    def get_shrinkage_fn(layer_name: str):
        if 'relu' in layer_name.lower():
            return
        if 'bn' in layer_name.lower():
            return ShrinkageManager.shrink_batch_norm
        if 'depthwise' in layer_name.lower():
            return ShrinkageManager.shrink_depthwise
        if 'project' in layer_name.lower():
            return ShrinkageManager.shrink_project_layer
        return

    @staticmethod
    def shrink_conv_layer(
        conv2d: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
    ) -> tf.keras.layers.Conv2D:
        kernel = conv2d.get_weights()[0]
        n_to_remove = len(indices_to_remove)

        assert conv2d.filters - n_to_remove > 0

        new_conv2d = copy_layer(conv2d, '', {'filters': conv2d.filters - n_to_remove})
        new_conv2d.build(conv2d.input.shape)

        indices_to_keep = [i for i in range(conv2d.filters) if i not in indices_to_remove]

        new_conv2d.set_weights([
            kernel[:, :, :, indices_to_keep]
        ])

        return new_conv2d

    @staticmethod
    def shrink_project_layer(
        project: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
        conv2d: tf.keras.layers.Conv2D,
    ) -> tf.keras.layers.Conv2D:

        kernel = project.get_weights()[0]
        new_project = copy_layer(project, '')
        shape = conv2d.output.shape.as_list()
        shape[-1] = shape[-1] - len(indices_to_remove)
        new_project.build(shape)

        indices_to_keep = [i for i in range(conv2d.filters) if i not in indices_to_remove]

        new_project.set_weights([
            kernel[:, :, indices_to_keep, :]
        ])

        return new_project

    @staticmethod
    def shrink_batch_norm(
        bn: tf.keras.layers.BatchNormalization,
        indices_to_remove: List[int],
        conv2d: tf.keras.layers.Conv2D,
    ) -> tf.keras.layers.BatchNormalization:
        new_bn = copy_layer(bn, '')
        shape = conv2d.output.shape.as_list()
        shape[-1] = conv2d.filters - len(indices_to_remove)
        new_bn.build(shape)

        indices_to_keep = [i for i in range(conv2d.filters) if i not in indices_to_remove]

        new_bn.set_weights([
            w[indices_to_keep] for w in bn.get_weights()
        ])

        return new_bn

    @staticmethod
    def shrink_depthwise(
        depthwise: tf.keras.layers.DepthwiseConv2D,
        indices_to_remove: List[int],
        conv2d: tf.keras.layers.Conv2D,
    ) -> tf.keras.layers.DepthwiseConv2D:
        new_depthwise = copy_layer(depthwise, '')
        shape = conv2d.output.shape.as_list()
        shape[-1] = conv2d.filters - len(indices_to_remove)
        new_depthwise.build(shape)

        indices_to_keep = [i for i in range(conv2d.filters) if i not in indices_to_remove]

        new_depthwise.set_weights([
            w[:, :, indices_to_keep, :] for w in depthwise.get_weights()
        ])

        return new_depthwise

    @staticmethod
    def shrink_decoder_layer(
        decoder: tf.keras.layers.Conv2D,
        indices_to_remove: List[int],
        conv2d: tf.keras.layers.Conv2D,
        from_zero: bool = True
    ):
        kernel = decoder.get_weights()[0]
        new_decoder = copy_layer(decoder)
        input_shape = decoder.input.shape.as_list()
        offset = input_shape[-1] - conv2d.filters
        input_shape[-1] = input_shape[-1] - len(indices_to_remove)
        new_decoder.build(input_shape)

        if from_zero:
            original_n = decoder.input.shape.as_list()[-1]
            indices_to_keep = list(range(conv2d.filters, original_n))
            indices_to_keep += [i for i in range(conv2d.filters) if i not in indices_to_remove]
        else:
            indices_to_keep = list(range(offset))
            indices_to_keep += [offset + i for i in range(conv2d.filters) if i not in indices_to_remove]

        new_decoder.set_weights([
            kernel[:, :, indices_to_keep, :]
        ])

        return new_decoder
