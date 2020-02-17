#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:24:38 2019

@author: Carlos Sánchez Muñoz
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Uso la notación Snake Case la cual es habitual en Python """

""" Lee una imagen ya sea en grises o en color
- file_name: archivo de la imagen
- flag_color: modo en el que se va a leer la imagen -> grises o color
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print('Leyendo ' + file_name + ' en gris')
    elif flag_color==1:
        print('Leyendo ' + file_name + ' en color')
    else:
        print('flag_color debe ser 0 o 1')

    img = cv2.imread(file_name, flag_color)
    return img
