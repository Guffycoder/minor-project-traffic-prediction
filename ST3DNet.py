# ST3DNet.py
"""
TensorFlow2 / Keras implementation of a simplified ST-3DNet encoder.
Designed to accept input shape (batch, nb_flow, len_closeness, H, W)
(i.e. channels_first with time as the 3rd axis).
Returns a tf.keras.Model whose output is (batch, out_channels, H, W).
"""

import tensorflow as tf
from tensorflow import layers, Model, backend as K

# Use channels_first to match your data representation (N, C, T, H, W)
K.set_image_data_format("channels_first")


class ReLUConv3D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding="same", data_format="channels_first"):
        super().__init__()
        self.conv3d = layers.Conv3D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    kernel_initializer="glorot_uniform")
        self.act = layers.ReLU()

    def call(self, x):
        x = self.conv3d(x)
        return self.act(x)


def residual_unit_2d(x, filters, data_format="channels_first"):
    """
    A simple 2D residual block applied on (batch, channels, H, W) tensors using Conv2D.
    Assumes input is (B, C, H, W) when data_format='channels_first'.
    """
    # convert to channels_last for Conv2D convenience or use data_format
    y = layers.BatchNormalization(axis=1)(x) if data_format == "channels_first" else layers.BatchNormalization()(x)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, kernel_size=3, padding="same", data_format=data_format)(y)
    y = layers.BatchNormalization(axis=1)(y) if data_format == "channels_first" else layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, kernel_size=3, padding="same", data_format=data_format)(y)
    # shortcut: if channels mismatch, project
    in_channels = x.shape[1] if data_format == "channels_first" else x.shape[-1]
    if int(in_channels) != int(filters):
        shortcut = layers.Conv2D(filters, kernel_size=1, padding="same", data_format=data_format)(x)
    else:
        shortcut = x
    return layers.Add()([shortcut, y])


def ST3DNet(c_conf=(6, 2, 16, 8), t_conf=None, external_dim=None, nb_residual_unit=4, out_channels=8):
    """
    Build a compact ST-3DNet-like encoder.
    Arguments:
      c_conf: tuple (len_closeness, nb_flow, map_height, map_width)
      t_conf: (unused here) kept for API compatibility
      external_dim: if provided, adds an external input path that is fused mechanically
      nb_residual_unit: number of 2D residual units after 3D->2D conv
      out_channels: number of output channels in final feature map (defaults 8)
    Returns:
      tf.keras.Model accepting inputs of shape (nb_flow, len_closeness, H, W) (channels_first)
      and producing (out_channels, H, W) per sample (channels_first).
    """
    len_closeness, nb_flow, map_height, map_width = c_conf

    # Input for closeness component
    inputs = []
    outputs = []

    if len_closeness and len_closeness > 0:
        # input shape: (nb_flow, len_closeness, H, W) with channels_first ordering
        inp_c = layers.Input(shape=(nb_flow, len_closeness, map_height, map_width), name="input_closeness")
        inputs.append(inp_c)

        # Conv3D stack (use data_format='channels_first' so input shape matches)
        # First 3D conv: aggregate along the time (depth) axis
        # Use kernel depth = min(6, len_closeness) to mimic original
        kd = min(6, len_closeness)
        x = layers.Conv3D(filters=64, kernel_size=(kd, 3, 3),
                          strides=(1, 1, 1), padding="same",
                          data_format="channels_first",
                          kernel_initializer="glorot_uniform")(inp_c)
        x = layers.Activation("relu")(x)

        # second 3D conv (downsample temporal dimension a bit with stride on depth)
        dstride = (max(1, len_closeness // 2), 1, 1)
        x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3),
                          strides=dstride, padding="same",
                          data_format="channels_first")(x)
        x = layers.Activation("relu")(x)

        # third 3D conv - further temporal compression
        x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3),
                          strides=(1, 1, 1), padding="same",
                          data_format="channels_first")(x)

        # Now reshape / squeeze time dimension to get a 2D feature map:
        # x shape: (B, filters, depth', H, W) -> collapse depth' into filters using conv reshape
        # Use a Conv3D with kernel size (depth',1,1) with 'valid' to collapse depth dimension to 1
        depth_dim = x.shape[2] if x.shape[2] is not None else None
        if depth_dim is not None:
            x = layers.Conv3D(filters=64, kernel_size=(int(depth_dim), 1, 1),
                              strides=(1, 1, 1), padding="valid", data_format="channels_first")(x)
            # result shape: (B, 64, 1, H, W) -> reshape to (B, 64, H, W)
            x = layers.Reshape((64, map_height, map_width))(x)
        else:
            # fallback safe reshape: reduce along axis 2 by averaging (if shape unknown)
            x = layers.Lambda(lambda z: K.mean(z, axis=2))(x)  # (B, filters, H, W)

        # apply nb_residual_unit residual 2D blocks
        y = x
        for _ in range(nb_residual_unit):
            y = residual_unit_2d(y, filters=64, data_format="channels_first")

        # small conv to reduce channels
        y = layers.Conv2D(filters=32, kernel_size=3, padding="same", data_format="channels_first")(y)
        y = layers.Activation("relu")(y)

        # final output map for closeness: produce out_channels feature maps
        out_c = layers.Conv2D(filters=out_channels, kernel_size=1, padding="same", data_format="channels_first")(y)
        outputs.append(out_c)

    # fusion: if only closeness present, that is the main_output
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # if other branches (e.g., t_conf) exist, they would be added here; for now just sum
        main_output = layers.Add()(outputs)

    # external component fusion (optional)
    if external_dim is not None and external_dim > 0:
        ext_in = layers.Input(shape=(external_dim,), name="external_input")
        inputs.append(ext_in)
        h = layers.Dense(10, activation="relu")(ext_in)
        h = layers.Dense(out_channels * map_height * map_width, activation="relu")(h)
        h = layers.Reshape((out_channels, map_height, map_width))(h)
        main_output = layers.Add()([main_output, h])

    # final activation
    main_output = layers.Activation("relu", name="main_output")(main_output)

    model = Model(inputs=inputs, outputs=main_output, name="ST3DNet_tf2")

    return model


# Convenience: allow direct import of class-like factory
if __name__ == "__main__":
    # quick test of model creation
    m = ST3DNet(c_conf=(6, 2, 16, 8), t_conf=None, external_dim=None, nb_residual_unit=2, out_channels=8)
    m.summary()
