# -*- coding: utf-8 -*-
import cv2
import numpy as np 

class ImageProcessor(object):
  """docstring for ImageProcessor"""
  def __init__(self, image_url):
    super(ImageProcessor, self).__init__()
    self.init(image_url)

  def init(self, image_url):
    self.original_image = cv2.imread(image_url, 0)

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
  def add_salt_and_pepper_noisy(self, image, prob):
    if (float(prob) < 0 or float(prob) > 1):
      raise ValueError('Rango debe estar en el rango 0.0 a 1.0')
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < float(prob)] = 255
    noisy[rnd > 1-float(prob)] = 0
    return noisy
  
  """
    Calculamos un filtro de media.
    Este filtro es equivalente a la implementación de opencv de
    cv2.blur(image, kernel).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def convolution(image, kernel, border_type=cv2.BORDER_REFLECT_101):
    result = np.zeros(image.shape, dtype=np.uint8)
    return result

  """
    Calculamos un filtro de mediana.
    Este filtro es equivalente a la implementación de opencv de
    cv2.medianFilter(image, size).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def median_filter(image, size, border_type=cv2.BORDER_REFLECT_101):
    return image


  """
    Calculamos un filtro de mediana propuesto por Zhang & Karim.
    Este filtro es equivalente a la implementación de opencv de
    cv2.medianFilter(image, size).
    Tener en cuenta que debe utilizarse un borde para la expansion de la imagen.
  """
  def median_filter_zk(image, size, K, border_type=cv2.BORDER_REFLECT_101):
    return image