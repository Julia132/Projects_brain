import cv2  # графический модуль opencv
import numpy as np  # модуль операций над матрицами
from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом

mypath_in = 'C:/Users/inet/Desktop/part_start'  # больные и здоровые в серых тонах
mypath_out = 'C:/Users/inet/Desktop/part_finish_0'  # результат обработки больных и здоровых вручную
mypath_late = 'C:/Users/inet/Desktop/part_start_late'  # результат обработки больных и здоровых вручную
mypath_no = 'C:/Users/inet/Desktop/no pathologies'  # здоровые по мнению кода
mypath_part_0 = 'C:/Users/inet/Desktop/part_0'
onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]
onlyfiles_out = [f for f in listdir(mypath_out) if isfile(join(mypath_out, f))]
img = np.empty(len(onlyfiles), dtype=object)
for n in range(98, len(onlyfiles)):
    img[n] = cv2.imread(join(mypath_in, onlyfiles[n]))
    gray = cv2.cvtColor(img[n], cv2.COLOR_BGR2GRAY)

    newImg = cv2.resize(gray, (512, 512))
    R = np.mean(newImg)
    std = np.std(newImg)
    standardized_images_out = ((newImg - R) / std) * 40 + 127
    blur = cv2.blur(standardized_images_out, (7, 7))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, 1)
    thresh = cv2.adaptiveThreshold(opening.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dilation = cv2.dilate(thresh, kernel, 1)
    erode = cv2.erode(dilation, kernel, 3)
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel, 5)
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(closing.shape, np.uint8)
    draw = cv2.drawContours(mask, contours, n, 255, 25)
    cv2.imwrite(join(mypath_part_0, onlyfiles[n]), draw)