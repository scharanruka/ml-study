import tensorflow as tf

# Check if a GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Check if it can perform a simple computation
if tf.config.list_physical_devices("GPU"):
    print("TensorFlow is using the GPU!")
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        print(tf.matmul(a, b))
else:
    print("GPU NOT FOUND. Check your drivers and CUDA paths.")
