#!/usr/bin/env python
# coding: utf-8

import requests

def httpreq(im = "0000.jpg"):

    #url = 'http://localhost:9696/classifier'
    url = 'http://supermac7.pythonanywhere.com/classifier'

    img = {
        "im" : im,
        }
    res = requests.post(url, json=img).json()
    print(f'Predictions: {res}')
    
    return res
