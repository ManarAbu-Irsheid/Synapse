import tensorflow as tf, pandas as pd

AUTOTUNE = tf.data.AUTOTUNE

def _load(path, label, size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, (size, size))
    img = tf.cast(img, tf.float32) / 255.0          # Normalization
    return img, tf.cast(label, tf.float32)

def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0), label

def make_dataset(csv_path, training=False, batch_size=64, size=64):
    df = pd.read_csv(csv_path)
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values))
    if training:
        ds = ds.shuffle(10000, seed=42)
    ds = ds.map(lambda p, l: _load(p, l, size), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)
