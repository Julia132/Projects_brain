import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом

def contour ():
    mypath_late = 'C:/Users/inet/Desktop/part_start'
    mypath_contour = 'C:/Users/inet/Desktop/contour'
    mypath_contour_final = 'C:/Users/inet/Desktop/contour_final'
    mypath_out = 'C:/Users/inet/Desktop/part_finish'
    onlyfiles_no = [f for f in listdir(mypath_late) if isfile(join(mypath_late, f))]
    onlyfiles_out = [f for f in listdir(mypath_out) if isfile(join(mypath_out, f))]
    img = np.empty(len(onlyfiles_no), dtype=object)
    img_f = np.empty(len(onlyfiles_out), dtype=object)
    for n in range(0, len(onlyfiles_no)):
        img[n] = cv2.imread(join(mypath_late, onlyfiles_no[n]))
        edges = cv2.Canny(img[n], 0, 255)
        cv2.imwrite(join(mypath_contour, onlyfiles_no[n]), edges)
    for n in range(0, len(onlyfiles_out)):
        img_f[n] = cv2.imread(join(mypath_out, onlyfiles_no[n]))
        edges = cv2.Canny(img[n], 0, 255)
        cv2.imwrite(join(mypath_contour, onlyfiles_no[n]), edges)
    return edges
edges = contour ()