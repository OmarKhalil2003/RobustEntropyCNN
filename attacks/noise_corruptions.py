import tensorflow as tf
import tensorflow.image as tfi

def gaussian_noise(x, std=0.1):
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=std)
    return tf.clip_by_value(x + noise, 0.0, 1.0)

def salt_pepper_noise(x, prob=0.05):
    rnd = tf.random.uniform(tf.shape(x))
    x_sp = tf.where(rnd < prob / 2, 0.0, x)
    x_sp = tf.where(rnd > 1 - prob / 2, 1.0, x_sp)
    return x_sp

def motion_blur(x, kernel_size=5):
    kernel = tf.ones((kernel_size, kernel_size, 1, 1))
    kernel = kernel / tf.reduce_sum(kernel)

    x_gray = tf.image.rgb_to_grayscale(x)
    x_blur = tf.nn.depthwise_conv2d(
        x_gray,
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME"
    )
    return tf.clip_by_value(tf.image.grayscale_to_rgb(x_blur), 0.0, 1.0)

def jpeg_compression(x, quality=50):
    x_uint8 = tf.image.convert_image_dtype(x, tf.uint8)
    x_jpeg = tf.map_fn(
        lambda img: tf.image.decode_jpeg(
            tf.image.encode_jpeg(img, quality=quality)
        ),
        x_uint8,
        fn_output_signature=tf.uint8
    )
    return tf.image.convert_image_dtype(x_jpeg, tf.float32)
