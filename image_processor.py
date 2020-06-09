# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import log10, sqrt  

class ImageProcessor(object):
  """docstring for ImageProcessor"""
  def __init__(self, image_url):
    super(ImageProcessor, self).__init__()
    self.init(image_url)

  def init(self, image_url):
    image = cv2.imread(image_url, 0)
    self.original_image = image.astype('float64')


  """
    Retornamos la imagen original
  """
  def get_original_image(self):
    return self.original_image

  """
    Agregamos ruido sal y pimienta en una imagen.
    Considerar realizar un copia de la imagen.
    image: should be one-channel image with pixels in [0, 1] range
    prob: probabildad (umbral) que controla el nivel de ruido en la imagen.
      Considerar el rango del umbral de [0, 1]
      prob de ruido de sal = prob
      prob de ruido de pimienta =  1 - prob
  """
  def add_salt_and_pepper_noise(self, image, prob):
    if (float(prob) < 0 or float(prob) > 1):
      raise ValueError('Rango debe estar en el rango 0.0 a 1.0')
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noise = image.copy()
    noise[rnd < float(prob)] = 255
    noise[rnd > 1-float(prob)] = 0
    return noise
  
  """
    Calculamos la convolucion de una matriz por su kernel
    Este filtro es equivalente a la implementación de opencv de
    cv2.filter2D(image, kernel).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def convolution(self, image, kernel, border_type=cv2.BORDER_REFLECT_101):
    result = cv2.filter2D(image,-1, kernel, borderType = border_type)

    return result

  """
    Calculamos un filtro de mediana.
    Este filtro es equivalente a la implementación de opencv de
    cv2.medianFilter(image, size).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def median_filter(self, image, size, border_type=cv2.BORDER_REFLECT_101):
    result = np.zeros(image.shape, dtype=np.float64)
    pad = int(size) // 2
    expanded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, border_type)

    i = 0
    while(i<image.shape[0]):
      j = 0
      while(j<image.shape[1]):
        sub_image = expanded_image[i:i+int(size), j:j+int(size)]
        result[i][j] = np.median(sub_image)
        j+=1
      i+=1
    return result

  """
    Calculamos un filtro de mediana propuesto por Zhang & Karim.
    Este filtro es equivalente a la implementación de opencv de
    cv2.medianFilter(image, size).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def median_filter_zk(self, image, size, K, border_type=cv2.BORDER_REFLECT_101):
    kernel_1 = np.array([
      [ 0,  0, 0,  0,  0], 
      [ 0,  0, 0,  0,  0], 
      [-1, -1, 4, -1, -1],
      [ 0,  0, 0,  0,  0],
      [ 0,  0, 0,  0,  0]], dtype=np.float64)

    kernel_2 = np.array([
      [ 0,  0, -1,  0,  0], 
      [ 0,  0, -1,  0,  0], 
      [ 0,  0,  4,  0,  0],
      [ 0,  0, -1,  0,  0],
      [ 0,  0, -1,  0,  0]], dtype=np.float64)

    kernel_3 = np.array([
      [-1,  0,  0,  0,  0], 
      [ 0, -1,  0,  0,  0], 
      [ 0,  0,  4,  0,  0],
      [ 0,  0,  0, -1,  0],
      [ 0,  0,  0,  0, -1]], dtype=np.float64)

    kernel_4 = np.array([
      [ 0,  0,  0,  0, -1], 
      [ 0,  0,  0, -1,  0], 
      [ 0,  0,  4,  0,  0],
      [ 0, -1,  0,  0,  0],
      [-1,  0,  0,  0,  0]], dtype=np.float64)

    conv_1 = self.convolution(image, kernel_1, border_type)
    conv_2 = self.convolution(image, kernel_2, border_type)
    conv_3 = self.convolution(image, kernel_3, border_type)
    conv_4 = self.convolution(image, kernel_4, border_type)

    

    result = np.zeros(image.shape, dtype=np.float64)
    pad = int(size) // 2
    expanded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, border_type)

    i = 0
    while(i<image.shape[0]):
      j = 0
      while(j<image.shape[1]):
        pixels = np.array([conv_1[i][j], conv_2[i][j],conv_3[i][j],conv_4[i][j]])
        min = np.amin(np.absolute(pixels))
        if(min > int(K)):
          sub_image = expanded_image[i:i+int(size), j:j+int(size)]
          result[i][j] = np.median(sub_image)
        else:
          result[i][j] = image[i][j]
        j+=1
      i+=1

    return result


  """
    Calculamos el Error Cuadratico Medio de Acuerdo entre dos matrices.
  """
  @staticmethod
  def calc_mse(original_image, stimated_image):
    mse = (np.square(original_image - stimated_image)).mean()
    return mse

  """
    Calculamos el Proporción Máxima de Señal a Ruido dado un mse
  """
  @staticmethod
  def calc_psnr(mse):
    psnr = 20 * log10(255 / sqrt(mse)) 
    return psnr


  """
    Calculamos el Error absoluto medio de Acuerdo entre dos matrices.
  """
  @staticmethod
  def calc_mae(original_image, stimated_image):
    mse = (original_image - stimated_image).mean()
    return mse