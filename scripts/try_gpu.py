import tensorflow as tf


# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {len(physical_devices)}")
    for gpu in physical_devices:
        print(f" - {gpu}")
    # Set memory growth to avoid allocation issues
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. The code will run on the CPU.")