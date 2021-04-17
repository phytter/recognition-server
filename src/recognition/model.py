from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def auc(y_true, y_pred):
  auc = tf.metrics.auc(y_true, y_pred)[1]
  keras.backend.get_session().run(tf.local_variables_initializer())
  return auc

class Recognizer():

  def __init__(self, path_model):
    self.classes = ['Lambari','hibrido_tambaqui_pirapitinga','Lambari_rosa','Carpas_coloridas_KOI','Cachara_pura','Tambatinga','Jundia_rosa','Carpa_capim','Tambaqui','Pintado_jundiara','Pacu']
    self.model = tf.keras.models.load_model(path_model, custom_objects={'auc': auc})

  def predict(self, file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    img = Image.fromarray(img.astype("uint8"))
    img = img.resize((299, 299))

    x = np.asarray(img) / 255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    previsao = self.model.predict(images)
    predict_label = str(np.array(self.classes)[np.argmax(previsao, axis=-1)].tolist()[0])
    score = str(np.amax(previsao))

    return predict_label, score

  def get_classes(self):
    classes = map(lambda x: x.replace("_", " "), self.classes)
    return list(classes)