import tensorflow as tf

def setup_gpu(memory_limit=None):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found. Running on CPU.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

            if memory_limit is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit
                    )]
                )
        print(f"GPU ready: {len(gpus)} device(s)")
    except RuntimeError as e:
        print("GPU setup error:", e)
