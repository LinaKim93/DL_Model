import tensorflow as tf


def LeNet(input_tensor):
    """
    Build a LeNet Model

    input_tensor: Input tensor of the Model

    model: return model
    """
    x = tf.keras.layers.Conv2D(
        filters=6, kernel_size=5, activation='relu', padding="same")(input_tensor)
    x = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.Conv2D(
        filters=120, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=out)
    return model


if __name__ == "__main__":
    input_shape = (28, 28, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = LeNet(input_tensor)
    model.summary()
    model.save('Lenet_model.h5')