# -*- coding: utf-8 -*-
from __future__ import print_function
from image_processor import ImageProcessor
from matplotlib import pyplot as plt
import cv2
import numpy as np 
import argparse
import os

"""
  Mostramos las imagenes en una grilla 2x2
  - Imagen Original
  - Imagen con ruido sal y pimienta (especificar en el titulo la probabilidad)
  - Imagen Obtenida con el Filtro de Mediana (Especificar en el titulo el tamaño de la ventana)
  - Imagen Obtenida con el Filtro de Mediana de Zhang y Karim (Especificar en el titulo el tamaño de la ventana + El valor K)
"""
def show_images(image_processor, prob_salt_pepper, k_threshold, median_window_size):
  original_image = image_processor.get_original_image()
  with_salt_pepper = image_processor.add_salt_and_pepper_noise(original_image, prob_salt_pepper)
  with_mediam_filter = image_processor.median_filter(with_salt_pepper, median_window_size)
  with_zk = image_processor.median_filter_zk(with_salt_pepper, median_window_size, k_threshold)

  plt.subplot(2, 2, 1), plt.imshow(original_image, cmap='gray', vmin=0, vmax=255), plt.title('Imagen Original')
  plt.subplot(2, 2, 2), plt.imshow(with_salt_pepper, cmap='gray', vmin=0, vmax=255), plt.title(" Imagen con Ruido SyP con P={0}".format(prob_salt_pepper))
  plt.subplot(2, 2, 3), plt.imshow(with_mediam_filter, cmap='gray', vmin=0, vmax=255), plt.title(" Imagen con filtro de mediana con ventana={0}".format(median_window_size))
  plt.subplot(2, 2, 4), plt.imshow(with_zk, cmap='gray', vmin=0, vmax=255), plt.title("Imagen con Filtro de Zhang y Karim con K={0} y ventana={1}".format(k_threshold, median_window_size))
  plt.show()

"""
  Mostramos un grafico con matplotlib con los errores calculados
  El estilo del grafico queda a criterio de los integrantes del trabajo.
"""
def show_errors(image_processor, prob_salt_pepper, k_threshold, median_window_size):

  original_image = image_processor.get_original_image()
  with_salt_pepper = image_processor.add_salt_and_pepper_noise(original_image, prob_salt_pepper)

  #rango de 50 a 300 con intervalos de 50
  t_range = np.arange(50., 300., 50)
  zk_results = np.array([image_processor.median_filter_zk(with_salt_pepper, median_window_size, T) for T in t_range])
  mse_results = [image_processor.calc_mse(original_image, zk_image) for zk_image in zk_results]
  psnr_results = [image_processor.calc_psnr(mse) for mse in mse_results]
  mae_results = [image_processor.calc_mae(original_image, zk_image) for zk_image in zk_results]
  print(mse_results)
  print(mae_results)

  plt.title("Medicion cuantitativa del Filtro ZyK. ruido={0} y ventana={1}".format(prob_salt_pepper, median_window_size))
  plt.plot(t_range, mse_results, '^--c', label='MSE')
  plt.plot(t_range, psnr_results, '--or',label='PSNR')
  plt.plot(t_range, mae_results, '--sg', label='MAE')
  plt.legend(loc = 'upper left')
  plt.xlabel('Umbral')
  plt.ylabel('Error')
  plt.show()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-i', '--image', required=True, help='Ubicación de la imagen')
  ap.add_argument('-g', '--show-image-grid', required=False, help='Mostramos una cuadricula', action='store_true')
  ap.add_argument('-p', '--prob-salt-pepper', required=False, help='Probabilidad de filtro sal y Pimienta')
  ap.add_argument('-k', '--k-threshold', required=False, help='Valor Umbral K para filtro')
  ap.add_argument('-s', '--media-size-window', required=False, help='Valor de ventana de la media')
  ap.add_argument('-e', '--show-error', required=False, help='Valor de ventana de la media', action='store_true')

  ap.set_defaults(show_image_grid=False)
  ap.set_defaults(show_error=False)
  ap.set_defaults(prob_salt_pepper=0.1)
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

  if (args['show_error']):
    show_errors(image_processor, prob_salt_pepper, k_threshold, median_window_size)
