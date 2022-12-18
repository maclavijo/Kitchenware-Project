#!/usr/bin/env python
# coding: utf-8

import requests

def httpreq(im = "0190.jpg"):

    #url = 'http://localhost:9696/classifier'
    url = 'http://supermac7.pythonanywhere.com/classifier'
    print(url)
    #user = {"im":"0000.jpg"}
    im = "https://media.istockphoto.com/id/520537085/photo/teaspoon-steel-isolated.jpg?s=612x612&w=0&k=20&c=_NTGCi03R-DNbjp2SCy4ABbKMwMYGkLFi5-kPvCqH6g="

    img = {
        #"im": "https://media.istockphoto.com/id/520537085/photo/teaspoon-steel-isolated.jpg?s=612x612&w=0&k=20&c=_NTGCi03R-DNbjp2SCy4ABbKMwMYGkLFi5-kPvCqH6g="
        #"im" : "https://t3.ftcdn.net/jpg/02/75/16/00/360_F_275160059_SkK5HApn4AduORNqJeZnhiN7AuMDGHeZ.jpg"
        "im": "0190.jpg"
        }
    print(img)
    res = requests.post(url, json=img).json()
    print(f'Predictions: {res}')
    
    return res





