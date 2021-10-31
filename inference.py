from tensorflow.keras.initializers import glorot_normal
import tensorflow as tf
import os
import cv2
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAvgPool2D, Activation, Conv2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as backend
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow.keras.callbacks
import math
import time
import matplotlib.pyplot as plt
import PIL
import PIL.Image


def resize_image(
    x, size_target=None, flg_keep_aspect=False,
    rate_scale=1.0, flg_random_scale=False
):

    # convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # calculate resize coefficients
    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c, = img.shape

    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]
    if size_target is None:
        size_heigth_target = size_height_img * rate_scale
        size_width_target = size_width_img * rate_scale

    coef_height = 1
    coef_width = 1
    if size_height_img < size_heigth_target:
        coef_height = size_heigth_target / size_height_img
    if size_width_img < size_width_target:
        coef_width = size_width_target / size_width_img

    # calculate coeffieient to match small size to target size
    # scale coefficient if specified
    low_scale = rate_scale
    if flg_random_scale:
        low_scale = 1.0
    coef_max = max(coef_height, coef_width) * \
        np.random.uniform(low=low_scale, high=rate_scale)

    # resize image
    size_height_resize = math.ceil(size_height_img*coef_max)
    size_width_resize = math.ceil(size_width_img*coef_max)

    # method_interpolation = cv2.INTER_LINEAR
    method_interpolation = cv2.INTER_CUBIC
    # method_interpolation = cv2.INTER_NEAREST

    if flg_keep_aspect:
        img_resized = cv2.resize(
            img, dsize=(size_width_resize,
                        size_height_resize), interpolation=method_interpolation
        )
    else:
        img_resized = cv2.resize(
            img, dsize=(
                int(size_width_target*np.random.uniform(
                    low=low_scale, high=rate_scale
                    )),
                int(size_heigth_target*np.random.uniform(
                    low=low_scale, high=rate_scale)
                    )
            ), interpolation=method_interpolation
        )
    return img_resized


def resize_images(images, **kwargs):
    max_images = len(images)
    for i in range(max_images):
        images[i] = resize_image(images[i], **kwargs)
    return images

# crop image at the center


def center_crop_image(x, size_target=(336, 336)):

    # convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c, = img.shape

    # crop image
    h_start = int((size_height_img - size_heigth_target) / 2)
    w_start = int((size_width_img - size_width_target) / 2)
    img_cropped = img[h_start:h_start+size_heigth_target,
                      w_start:w_start+size_width_target, :]

    return img_cropped
# crop image of fixed-size from random point of top-left corner


def random_crop_image(x, size_target=(336, 336)):

    # convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c, = img.shape

    # crop image
    margin_h = (size_height_img - size_heigth_target)
    margin_w = (size_width_img - size_width_target)
    h_start = 0
    w_start = 0
    if margin_h != 0:
        h_start = np.random.randint(low=0, high=margin_h)
    if margin_w != 0:
        w_start = np.random.randint(low=0, high=margin_w)
    img_cropped = img[h_start:h_start+size_heigth_target,
                      w_start:w_start+size_width_target, :]

    return img_cropped
# flip image horizontally


def horizontal_flip_image(x):
    if np.random.random() >= 0.5:
        return x[:, ::-1, :]
    else:
        return x


# feature-wise normalization
def normalize_image(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):

    x = np.asarray(x, dtype=np.float32)

    if len(x.shape) == 4:
        for dim in range(3):
            x[:, :, :, dim] = (x[:, :, :, dim] - mean[dim]) / std[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[:, :, dim] = (x[:, :, dim] - mean[dim]) / std[dim]

    return x


def check_images(images):
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])

    plt.show()


class DirectoryIterator(keras.preprocessing.image.DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=backend.floatx())
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=None,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y


class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=16, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):

        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


def load_data(
    path_data_train=None, path_data_valid=None,
    size_width=336, size_heigth=336, size_mini_batch=16,
    flg_debug=False, **kwargs
):

    # set preprocessing functions
    def func_train(x): return normalize_image(
        random_crop_image(
            horizontal_flip_image(
                resize_image(x, size_target=(
                    size_heigth, size_width), flg_keep_aspect=True)
            )
        ), mean=[123.82988033, 127.3509729, 110.25606303]
    )

    def func_valid(x): return normalize_image(
        center_crop_image(
            resize_image(x, size_target=(size_heigth, size_width),
                         flg_keep_aspect=True)
        ), mean=[123.82988033, 127.3509729, 110.25606303]
    )

    # set image_data_generator
    gen_train = ImageDataGenerator(
        preprocessing_function=func_train
    )

    gen_valid = ImageDataGenerator(
        preprocessing_function=func_valid
    )

    gen_dir_train = gen_train.flow_from_directory(
        path_data_train, target_size=(
            size_heigth, size_width), batch_size=size_mini_batch
    )

    gen_dir_valid = gen_valid.flow_from_directory(
        path_data_valid, target_size=(
            size_heigth, size_width), batch_size=size_mini_batch, shuffle=False
    )

    return gen_dir_train, gen_dir_valid


train_dir = 'train_valid/train'
valid_dir = 'train_valid/val'
test_dir = 'testimg/'

gen_dir_train, gen_dir_test = load_data(
    path_data_train=train_dir, path_data_valid=test_dir, size_mini_batch=9
)

x_test, y_test = gen_dir_test.next()
check_images(x_test)


def outer_product(x):
    """
    calculate outer-products of 2 tensors
    args
    x
    list of 2 tensors
    , assuming each of which has shape
    =(size_minibatch, total_pixels, size_filter)
    """
    return keras.backend.batch_dot(
        x[0], x[1], axes=[1, 1]
    ) / x[0].get_shape().as_list()[1]


def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    temp = keras.backend.abs(x) + 1e-9
    return keras.backend.sign(x) * keras.backend.sqrt(temp)


def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args
            x
                a tensor
    """
    return keras.backend.l2_normalize(x, axis=axis)


def build_model(
    size_heigth=336, size_width=336, no_class=200,
    no_last_layer_backbone=17, name_optimizer="sgd", rate_learning=1.0,
    rate_decay_learning=0.0, rate_decay_weight=0.0,
    name_initializer="glorot_normal",
    name_activation_logits="softmax", name_loss="categorical_crossentropy",
    flg_debug=False, **kwargs
):

    keras.backend.clear_session()

    print("-------------------------------")
    print("parameters:")
    for key, val in locals().items():
        if val is not None and not key == "kwargs":
            print("\t", key, "=",  val)
    print("-------------------------------")

    ###
    # load pre-trained model
    ###
    tensor_input = keras.layers.Input(shape=[size_heigth, size_width, 3])
    model_detector = keras.applications.vgg16.VGG16(
        input_tensor=tensor_input, include_top=False, weights='imagenet'
    )

    ###
    # bi-linear pooling
    ###

    # extract features from detector
    x_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape
    if flg_debug:
        print("shape_detector : {}".format(shape_detector))

    # extract features from extractor ,
    # same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print("shape_extractor : {}".format(shape_extractor))

    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
        [
            shape_detector[1] * shape_detector[2], shape_detector[-1]
        ]
    )(x_detector)
    if flg_debug:
        print("x_detector shape after ops : {}".format(x_detector.shape))

    x_extractor = keras.layers.Reshape(
        [
            shape_extractor[1] * shape_extractor[2], shape_extractor[-1]
        ]
    )(x_extractor)
    if flg_debug:
        print("x_extractor shape after ops : {}".format(x_extractor.shape))

    # outer products of features,
    # output shape=(minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    if flg_debug:
        print("x shape after outer products ops : {}".format(x.shape))

    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    if flg_debug:
        print("x shape after rehsape ops : {}".format(x.shape))

    # signed square-root
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print("x shape after signed-square-root ops : {}".format(x.shape))

    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    if flg_debug:
        print("x shape after L2-Normalization ops : {}".format(x.shape))

    ###
    # attach FC-Layer
    ###

    if name_initializer is not None:
        name_initializer = eval(name_initializer+"()")

    x = keras.layers.Dense(
        units=no_class,
        kernel_regularizer=keras.regularizers.l2(rate_decay_weight),
        kernel_initializer=name_initializer
    )(x)
    if flg_debug:
        print("x shape after Dense ops : {}".format(x.shape))
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)
    if flg_debug:
        print("prediction shape : {}".format(tensor_prediction.shape))

    ###
    # compile model
    ###
    model_bilinear = keras.models.Model(
        inputs=[tensor_input], outputs=[tensor_prediction]
    )
    # fix pre-trained weights
    for layer in model_detector.layers:
        layer.trainable = False

    # define optimizers
    opt_adam = tf.keras.optimizers.Adam(
        lr=rate_learning, decay=rate_decay_learning
    )
    opt_rms = tf.keras.optimizers.RMSprop(
        lr=rate_learning, decay=rate_decay_learning
    )
    opt_sgd = tf.keras.optimizers.SGD(
        lr=rate_learning, decay=rate_decay_learning,
        momentum=0.9, nesterov=False
    )
    optimizers = {
        "adam": opt_adam, "rmsprop": opt_rms, "sgd": opt_sgd
    }

    model_bilinear.compile(
        loss=name_loss, optimizer=optimizers[name_optimizer], metrics=[
            "categorical_accuracy"]
    )

    if flg_debug:
        model_bilinear.summary()

    return model_bilinear


model = build_model(
    # number of output classes, 200 for CUB200
    no_class=200,  # pretrained model specification, using VGG16
    # "block5_conv3 "
    no_last_layer_backbone=17,          # training parametes
    rate_learning=1.0, rate_decay_weight=1e-8, flg_debug=True
)

for layer in model.layers:
    layer.trainable = True

model.load_weights('model/BCNN_keras/1031_3/E[30]_LOS[1.395]_ACC[0.667].h5')

opt_sgd = tf.keras.optimizers.SGD(
    lr=1e-3, decay=1e-9, momentum=0.9, nesterov=False
)
model.compile(
    loss="categorical_crossentropy", optimizer=opt_sgd,
    metrics=["categorical_accuracy"]
)

pred = model.predict(gen_dir_test, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

print(gen_dir_test.filenames[-1])

predicted_class_indices = np.argmax(pred, axis=1)

labels = (gen_dir_train.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# print(predictions)

filenames = gen_dir_test.filenames
dicts = {}
keys = []
predicts = []
for idx in range(len(filenames)):
    # print('predict  %s' % ((predictions[idx]))) #001.Black_footed_Albatross
    predicts.append(predictions[idx])
    filenamestr = filenames[idx]
    keys.append(filenamestr[-8:])
    print('title    %s' % filenamestr[-8:])  # 5565.jpg
    # print('')

# 根據圖片名稱建dictionary dict[i]=predict class to i.jpg
# read order => 搜尋dictionary

for i in range(len(keys)):
    dicts[keys[i]] = predicts[i]

# fname='3212.jpg'
# print(fname+' '+dicts[fname])

f = open('testing_img_order.txt', 'r')
testorder = []

for line in f.readlines():
    order = line.split('\n')[0]
    testorder.append(order)
f.close()

f = open('model/BCNN_keras/1031_3/answer.txt', 'w')
for i in range(len(testorder)):
    fname = testorder[i]
    f.write(fname+' '+dicts[fname]+'\n')
f.close()
