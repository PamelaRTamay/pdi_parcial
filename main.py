# -*- coding: utf-8 -*-
from __future__ import print_function
from image_processor import ImageProcessor
from matplotlib import pyplot as plt
import cv2
import numpy as np 
import argparse
import os

def show_images(image_processor, prob_salt_pepper, k_threshold, median_window_size):
  plt.subplot(2, 2, 1), plt.imshow(image_processor.get_original_image(), cmap='gray', vmin=0, vmax=255), plt.title('Original')
  plt.subplot(2, 2, 2), plt.imshow(image_processor.add_salt_and_pepper_noisy(image_processor.get_original_image(), prob_salt_pepper), cmap='gray', vmin=0, vmax=255), plt.title('Original')
  #plt.subplot(2, 2, 2), plt.imshow(image_processor.add_salt_and_pepper_noisy(image_processor.get_original_image(), prob_salt_pepper), cmap='gray', vmin=0, vmax=255), plt.title('Original')
  plt.show()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--image', required=True, help='Ubicación de la imagen')
  ap.add_argument('-g', '--show-image-grid', required=False, help='Mostramos una cuadricula', action='store_true')
  ap.add_argument('-p', '--prob-salt-pepper', required=False, help='Probabilidad de filtro sal y Pimienta')
  ap.add_argument('-k', '--k-threshold', required=False, help='Valor Umbral K para filtro')
  ap.add_argument('-s', '--media-size-window', required=False, help='Valor de ventana de la media')

  ap.set_defaults(show_image_grid=False)
  ap.set_defaults(prob_salt_pepper=0.5)
  ap.set_defaults(k_threshold=100)
  ap.set_defaults(media_size_window=3)

  args = vars(ap.parse_args())

  image_url = args['image']
  prob_salt_pepper = args['prob_salt_pepper']
  k_threshold = args['k_threshold']
  median_window_size = args['media_size_window']

  image_processor = ImageProcessor(image_url)

  if (args['show_image_grid']):
    show_images(image_processor, prob_salt_pepper, k_threshold, median_window_size)
