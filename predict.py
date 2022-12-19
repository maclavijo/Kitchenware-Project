
import json
import numpy as np
from PIL import Image
import requests
from keras.models import load_model
from keras.applications.xception import preprocess_input

from flask import Flask
from flask import request

app = Flask('kitchenware')

@app.route('/classifier', methods=['POST'])

def classifier():
    
    PIXELS = 300
    model = load_model('Models/best_Xception_lr0.001_09_0.960.h5')

    inputImg = request.get_json()
    print('here:',inputImg)
    print(inputImg['im'])
    res = []
    classes = {'cup': 0, 'fork': 1, 'glass': 2, 'knife': 3, 'plate': 4, 'spoon': 5}

    labels = dict((v, k) for k, v in classes.items())

    im = Image.open(requests.get(inputImg['im'], stream=True).raw)
    #im = Image.open(f'{inputImg["im"]}')
    print(im)
    im = im.resize((PIXELS, PIXELS  ))

    x = np.array(im)
    X = np.array([x])

    res = []
    X = preprocess_input(X)
    preds = np.round(model.predict(X),4) * 100
    label = preds[0]
    print(label)
    res.append(label)
    results = {i:round(float(j),2) for i,j in zip(classes.keys(),label)}

    print(results)
    return json.dumps(results)
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




