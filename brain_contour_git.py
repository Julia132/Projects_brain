import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом

mypath_no = 'C:/Users/inet/Desktop/no pathologies'
mypath_contour = 'C:/Users/inet/Desktop/contour'
onlyfiles_no = [f for f in listdir(mypath_no) if isfile(join(mypath_no, f))]
onlyfiles_contour = [f for f in listdir(mypath_contour) if isfile(join(mypath_contour, f))]
img = np.empty(len(onlyfiles_no), dtype=object)
for n in range(0, len(onlyfiles_no)):
    img[n] = cv2.imread(join(mypath_no, onlyfiles_no[n]))
    edges = cv2.Canny(img[n], 0, 255)
    cv2.imwrite(join(mypath_contour, onlyfiles_no[n]), edges)
