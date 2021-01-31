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

model = tf.keras.models.load_model('./mydata_modelo_1_argumented.h5', custom_objects={'auc': auc})

@app.route('/predict' , methods=['POST'])
def mask_image():

	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

	img = Image.fromarray(img.astype("uint8"))
	img = img.resize((299, 299))
	# rawBytes = io.BytesIO()
	# img.save(rawBytes, "JPEG")
	# rawBytes.seek(0)
	# img_base64 = base64.b64encode(rawBytes.read())

	x = np.asarray(img) / 255
	x = np.expand_dims(x, axis=0)
	
	images = np.vstack([x])
	previsao = model.predict(images)

	return jsonify({'label':str(np.array(classes)[np.argmax(previsao, axis=-1)].tolist()[0]), 'score': str(np.amax(previsao))})

@app.route("/hello", methods=["GET"])
def hello_world():
    return {"hello": "World"}

# @app.after_request
# def after_request(response):
#     print("log: setting cors" , file = sys.stderr)
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')