import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50V2, VGG19, EfficientNetB1, InceptionV3, ConvNeXtBase
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dropout, Dense, SeparableConv2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from ultralytics import YOLO


from modules.config import EMOTIONS
from modules.loss import categorical_focal_loss


class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

def build_model_final_layers(learning_rate, dropout_rate, l2_reg, initial_bias, model_name='PattLite'):
    NUM_CLASSES = 7
    IMG_SHAPE = (128, 128, 3)
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    #sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")

    # Seleziona il modello backbone
    if model_name == 'EfficientNetB1':
        backbone = EfficientNetB1(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('block5c_add').output, name='base_model')
    elif model_name == 'VGG19':
        backbone = VGG19(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('block4_pool').output, name='base_model')
    elif model_name == 'PattLite':
        backbone = MobileNet(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        base_model = Model(backbone.input, backbone.layers[-29].output, name='base_model')
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    elif model_name == 'ResNet':
        backbone = ResNet50V2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('conv4_block5_out').output, name='base_model')
    elif model_name == 'ConvNeXt':
        backbone = tf.keras.applications.ConvNeXtBase(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(128,128,3),
        classifier_activation='softmax'
        )
        base_model = Model(backbone.input, backbone.get_layer('convnext_base_stage_2_block_24_identity').output, name='base_model')
    elif model_name == 'InceptionV3':
        backbone = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        base_model = Model(backbone.input, backbone.get_layer('mixed5').output, name='base_model')

    else:
        raise ValueError(f"Modello '{model_name}' non supportato.")


    base_model.trainable = False

    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
    patch_extraction = tf.keras.Sequential([
        SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(l2_reg))
    ], name='patch_extraction')

    global_average_layer = GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
                                              BatchNormalization()], name='pre_classification')
    prediction_layer = Dense(NUM_CLASSES, activation="softmax", name='classification_head', bias_initializer=Constant(initial_bias))

    x = input_layer
    if model_name != 'ConvNeXt':
        x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = global_average_layer(x)
    x = Dropout(dropout_rate)(x)
    x = pre_classification(x)
    x = ExpandDimsLayer(axis=-1)(x)
    x = self_attention([x, x])
    x = SqueezeLayer(axis=-1)(x)
    outputs = prediction_layer(x)

    model = Model(inputs=input_layer, outputs=outputs, name='train-head')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                metrics=['categorical_accuracy'])

    return model

def load_model(model_name, model_path_subset):
    custom_objects = {'loss': categorical_focal_loss()}

    if not model_name in model_path_subset.keys():
        raise ValueError(f"Model name '{model_name}' not found in the provided model path subset.")

    if "efficientnet" in model_name:
        return load_model_efficientnet(model_name, model_path_subset, custom_objects)

    if "yolo" in model_name:
        return YOLO("/content/drive/MyDrive/Colab Notebooks/HPC/finale/yolov8n/yolov8n3/weights/last.pt")
    
    # For other models. Probably: resnet, vgg, inception, convnext, pattlite
    with keras.utils.custom_object_scope(custom_objects):
        # One of these three should work:
        # model = keras.layers.TFSMLayer(model_path_subset[model_name], call_endpoint='serve')
        # model = keras.layers.TFSMLayer(model_path_subset[model_name], call_endpoint='serving_default')
        model = keras.models.load_model(model_path_subset[model_name])
        return model
    
def load_model_efficientnet(model_name, model_path_subset, custom_objects):
    with tf.keras.utils.custom_object_scope(custom_objects):
        # Genera un bias iniziale casuale per ciascuna classe
        class_names = EMOTIONS
        num_classes = len(class_names)
        initial_bias = np.random.randn(num_classes)
        model = build_model_final_layers(0.001, 0.1, 0.1, initial_bias, 'EfficientNetB1')
        backbone = model.get_layer('base_model')
        backbone.trainable = True
        unfreeze = 114
        fine_tune_from = len(backbone.layers) - unfreeze
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self_attention = model.get_layer('attention')
        patch_extraction = model.get_layer('patch_extraction')
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        global_average_layer = model.get_layer('gap')
        prediction_layer = model.get_layer('classification_head')
        IMG_SHAPE = (128, 128, 3)
        input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
        #sample_resizing = tf.keras.layers.Resizing(128, 128, name="resize")
        l2_reg = 0.07099871122599184
        learning_rate = 0.0005486860365638318
        dropout_rate = 0.4603464152900125
        x = input_layer
        pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu', kernel_regularizer = l2(l2_reg)),
                                            tf.keras.layers.BatchNormalization()], name='pre_classification')

        x = preprocess_input(x)
        x = backbone(x, training=False)
        x = patch_extraction(x)
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
        x = global_average_layer(x)
        x = Dropout(dropout_rate)(x)
        x = pre_classification(x)
        x = ExpandDimsLayer(axis=-1)(x)
        x = self_attention([x, x])
        x = SqueezeLayer(axis=-1)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = prediction_layer(x)

        efficientnet = Model(inputs=input_layer, outputs=outputs, name='train-head')
        efficientnet.summary(show_trainable=True)
        efficientnet.load_weights(model_path_subset[model_name])
        efficientnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=3.0),
                loss= categorical_focal_loss(alpha=0.25, gamma=2.0),
                metrics=['categorical_accuracy'])
        
        return efficientnet