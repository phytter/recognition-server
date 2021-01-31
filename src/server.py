from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

app = Flask("Recogntion")
classes = [b'Zanclus_cornutus',b'Platy_sangue',b'Pomacentrus_moluccensis',b'Chaetodon_lunulatus',b'Acara_bandeira',b'Tricogaster',b'Pacu',b'Shark',b'Zebrasoma_scopas',b'Dourado',b'Scaridae',b'Lampris_guttatus',b'Chaetodon_trifascialis',b'Chromis_chrysura',b'Paulistinha',b'Pempheris_vanicolensis',b'Amphiprion_clarkii',b'Balistapus_undulatus',b'Pintado',b'Tambaqui',b'NoF',b'Siganus_fuscescens',b'Barbus_ouro',b'Bigeye_tuna',b'Kinguio_korraco',b'hibrido_tambaqui_pirapitinga',b'Acara_disco',b'Tambatinga',b'Piau_tres_pintas',b'Papagaio',b'Tucunare',b'Kinguio_cometa_calico',b'Canthigaster_valentini',b'Plectroglyphidodon_dickii',b'Neoglyphidodon_nigroris',b'Hemigymnus_melapterus',b'Scolopsis_bilineata',b'Lambari',b'Platy_rubi',b'Kinguio',b'Dolphinfish',b'Neoniphon_sammara',b'Telescopio',b'Mato_grosso',b'Oscar',b'Palhaco',b'Oscar_albino',b'Albacore_tuna',b'Yellowfin_tuna',b'Hemigymnus_fasciatus',b'Beta',b'Myripristis_kuntee',b'Platy_laranja',b'Carpa_media',b'Barbus_sumatra',b'Acanthurus_nigrofuscus',b'Molinesia_preta',b'Abudefduf_vaigiensis',b'Acara_bandeira_marmorizado',b'Dascyllus_reticulatus',b'Carpa',b'Lutjanus_fulvus',b'Tetra_negro']

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

model = tf.keras.models.load_model('./master_v4_modelo_1.h5', custom_objects={'auc': auc})


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

	return jsonify({'results':str(np.array(classes)[np.argmax(previsao, axis=-1)].tolist())})

@app.route("/", methods=["GET"])
def hello_world():
    return {"hello": "World"}

@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')