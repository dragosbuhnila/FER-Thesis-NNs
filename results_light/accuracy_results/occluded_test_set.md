Final evaluation results on occluded test set:  
Model: resnet_finetuning - Test Loss: 0.2720, Test Accuracy: 0.4543  
Model: pattlite_finetuning - Test Loss: 0.3897, Test Accuracy: 0.3543  
Model: vgg19_finetuning - Test Loss: 0.3715, Test Accuracy: 0.4314  
Model: inceptionv3_finetuning - Test Loss: 0.3303, Test Accuracy: 0.4314  
Model: convnext_finetuning - Test Loss: 0.3351, Test Accuracy: 0.4114 





<!-- ==================================================================================== -->





complete output:  
2025-10-15 18:11:17.648728: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

lts due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

_OPTS=0`.
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

es.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

Generator initialized: test mode
======================================
Evaluating model: resnet_finetuning
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\saving\legacy\saved_model\load.py:107: The name tf.gfile.Exists is deprecated. Please use tf.io.gfile.exists instead.

2025-10-15 18:11:49.785941: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\engine\functional.py:156: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\layers\pooling\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

6/6 [==============================] - 5s 576ms/step - loss: 0.2720 - categorical_accuracy: 0.4543
======================================
Evaluating model: pattlite_finetuning
6/6 [==============================] - 2s 146ms/step - loss: 0.3897 - categorical_accuracy: 0.3543
======================================
Evaluating model: vgg19_finetuning
6/6 [==============================] - 16s 3s/step - loss: 0.3715 - categorical_accuracy: 0.4314
======================================
Evaluating model: inceptionv3_finetuning
6/6 [==============================] - 3s 253ms/step - loss: 0.3303 - categorical_accuracy: 0.4314
======================================
Evaluating model: convnext_finetuning
2025-10-15 18:13:21.750455: I external/local_xla/xla/service/service.cc:168] XLA service 0x1c6815d1470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2025-10-15 18:13:21.750654: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\engine\functional.py:156: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\layers\pooling\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From C:\Users\Dragos\Roba\Lectures\YM2.2\Thesis\e Models\.venv\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

6/6 [==============================] - 5s 576ms/step - loss: 0.2720 - categorical_accuracy: 0.4543
======================================
Evaluating model: pattlite_finetuning
6/6 [==============================] - 2s 146ms/step - loss: 0.3897 - categorical_accuracy: 0.3543
======================================
Evaluating model: vgg19_finetuning
6/6 [==============================] - 16s 3s/step - loss: 0.3715 - categorical_accuracy: 0.4314
======================================
Evaluating model: inceptionv3_finetuning
6/6 [==============================] - 3s 253ms/step - loss: 0.3303 - categorical_accuracy: 0.4314
======================================
Evaluating model: convnext_finetuning
2025-10-15 18:13:21.750455: I external/local_xla/xla/service/service.cc:168] XLA service 0x1c6815d1470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2025-10-15 18:13:21.750654: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
======================================
Evaluating model: pattlite_finetuning
6/6 [==============================] - 2s 146ms/step - loss: 0.3897 - categorical_accuracy: 0.3543
======================================
Evaluating model: vgg19_finetuning
6/6 [==============================] - 16s 3s/step - loss: 0.3715 - categorical_accuracy: 0.4314
======================================
Evaluating model: inceptionv3_finetuning
6/6 [==============================] - 3s 253ms/step - loss: 0.3303 - categorical_accuracy: 0.4314
======================================
Evaluating model: convnext_finetuning
2025-10-15 18:13:21.750455: I external/local_xla/xla/service/service.cc:168] XLA service 0x1c6815d1470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2025-10-15 18:13:21.750654: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
Evaluating model: inceptionv3_finetuning
6/6 [==============================] - 3s 253ms/step - loss: 0.3303 - categorical_accuracy: 0.4314
======================================
Evaluating model: convnext_finetuning
2025-10-15 18:13:21.750455: I external/local_xla/xla/service/service.cc:168] XLA service 0x1c6815d1470 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2025-10-15 18:13:21.750654: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
s not guarantee that XLA will be used). Devices:
2025-10-15 18:13:21.750654: I external/local_xla/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
I0000 00:00:1760544806.153859   23436 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
 the process.
2025-10-15 18:13:26.183972: E external/local_xla\xla/stream_executor/stream_executor_internal.h:177] SetPriority unimplemented for this stream.
6/6 [==============================] - 183s 28s/step - loss: 0.3351 - categorical_accuracy: 0.4114
======================================
Final evaluation results on occluded test set:
Model: resnet_finetuning - Test Loss: 0.2720, Test Accuracy: 0.4543
Model: pattlite_finetuning - Test Loss: 0.3897, Test Accuracy: 0.3543
Model: vgg19_finetuning - Test Loss: 0.3715, Test Accuracy: 0.4314
Model: inceptionv3_finetuning - Test Loss: 0.3303, Test Accuracy: 0.4314
Model: convnext_finetuning - Test Loss: 0.3351, Test Accuracy: 0.4114