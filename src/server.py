from flask import Flask, render_template , request , jsonify
from flask_cors import CORS
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

app = Flask("Recogntion")
CORS(app)
classes = ['Lambari','hibrido_tambaqui_pirapitinga','Lambari_rosa','Carpas_coloridas_KOI','Cachara_pura','Tambatinga','Jundia_rosa','Carpa_capim','Tambaqui','Pintado_jundiara','Pacu']

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

model = tf.keras.models.load_model('./model.h5', custom_objects={'auc': auc})

@app.route('/predict' , methods=['POST'])
def predict_image():

	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

	img = Image.fromarray(img.astype("uint8"))
	img = img.resize((299, 299))

	x = np.asarray(img) / 255
	x = np.expand_dims(x, axis=0)
	
	images = np.vstack([x])
	previsao = model.predict(images)
	predict_label = str(np.array(classes)[np.argmax(previsao, axis=-1)].tolist()[0])
	score = str(np.amax(previsao))
	
	return jsonify({'label': predict_label, 'score': score})

@app.route("/", methods=["GET"])
def checkHealth():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')