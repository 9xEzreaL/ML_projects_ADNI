import os

import cv2
import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications.vgg16 import preprocess_input
from matplotlib import pyplot as plt
import numpy as np

def watch_layer(layer, tape):
    """
    Make an intermediate hidden `layer` watchable by the `tape`.
    After calling this function, you can obtain the gradient with
    respect to the output of the `layer` by calling:

        grads = tape.gradient(..., layer.result)

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result of `layer.call` internally.
            layer.result = func(*args, **kwargs)
            # From this point onwards, watch this tensor.
            tape.watch(layer.result)
            # Return the result to continue with the forward pass.
            return layer.result
        return wrapper
    layer.call = decorator(layer.call)
    return layer

def processing_image(img_path):
    # 讀取影像為 PIL 影像
    img = image.load_img(img_path, target_size=(224, 224))

    # 轉換 PIL 影像為 nparray
    x = image.img_to_array(img)
    # 加上一個 batch size，例如轉換 (224, 224, 3) 為 （1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    # 將 RBG 轉換為 BGR，並解減去各通道平均
    x = preprocess_input(x)
    return x


def gradcam(model, x):
    tf.executing_eagerly()
    # 取得影像的分類類別
    preds = model.predict(x) # [[0.,1.]]
    pred_class = np.argmax(preds[0])

    # 取得影像分類名稱
    # pred_class_name = imagenet_utils.decode_predictions(preds)[0][0][1]

    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer('vgg16').get_layer("block5_conv3")

    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    with tf.GradientTape() as gtape:
        watch_layer(last_conv_layer, gtape)

        model_prediction = model.output[:, np.argmax(preds[0])]
        grads = gtape.gradient(model_prediction, last_conv_layer.output)

    # grads = K.gradients(pred_output, last_conv_layer.output)[0]
    # 求得針對每個 feature map 的梯度加總
    pooled_grads = K.sum(grads, axis=(0, 1, 2))

    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output])

    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, i] *= (pooled_grads_value[i])

    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    return heatmap, preds


def plot_heatmap(heatmap, img_path, preds):
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    # 正規化
    heatmap /= np.max(heatmap)
    # 讀取影像
    img = cv2.imread(img_path)
    fig, ax = plt.subplots()
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))

    # 拉伸 heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    # 以 0.6 透明度繪製原始影像
    ax.imshow(im, alpha=0.6)

    # 以 0.4 透明度繪製熱力圖
    ax.imshow(heatmap, cmap='jet', alpha=0.4)

    plt.title(preds)

    plt.show()

if __name__=='__main__':
    weight_file_dir = 'dataset_2class/alzheimer_inception_cnn_model'
    model = keras.models.load_model(weight_file_dir,custom_objects={'F1Score':tfa.metrics.F1Score(num_classes=2)})
    img_path = 'ADNI_2class/AD/91.png'
    img = processing_image(img_path)

    heatmap, pred_class_name = gradcam(model, img)

    plot_heatmap(heatmap, img_path, pred_class_name)