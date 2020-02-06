import cv2
import numpy as np
import csv
import math

from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом
from math import pi  # для константы pi

#from scipy.stats import kurtosis
#from scipy.stats import skew

def dataset():
    mypath_in = 'C:/Users/inet/Desktop/part_start'
    mypath_late = 'C:/Users/inet/Desktop/part_start_late'
    onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]
    onlyfiles_out = [f for f in listdir(mypath_late) if isfile(join(mypath_late, f))]
    img = np.empty(len(onlyfiles_out), dtype=object)

    with open('data_set_new.csv', 'w', newline='') as f:
            wr = csv.writer(f, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
            for n in range(0, len(onlyfiles)):

                img[n] = cv2.imread(join(mypath_in, onlyfiles[n]))
                gray = cv2.cvtColor(img[n], cv2.COLOR_BGR2GRAY)
                newImg = cv2.resize(gray, (512, 512))
                R = np.mean(newImg)
                std = np.std(newImg)

                img[n] = cv2.imread(join(mypath_late, onlyfiles[n]))
                gray = cv2.cvtColor(img[n], cv2.COLOR_BGR2GRAY)

                closing = cv2.resize(gray, (512, 512))

                height, width = closing.shape
                ret, threshold_image = cv2.threshold(closing, 127, 255, 0)

                _, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 1:
                    print("WARNING! len(contours) == ", len(contours))
                    for contour in contours:
                        print(len(contour))
                contours.sort(key=lambda contour: len(contour), reverse=True)
                if len(contours) == 0:
                    continue
                i = 0
                area = cv2.contourArea(contours[i])
                hull = cv2.convexHull(contours[i])
                hull_area = cv2.contourArea(hull)

                _, radius = cv2.minEnclosingCircle(contours[i])
                radius = int(radius)

                if 0 < radius:

                    square = area / (width*height)
                    rh = radius / height
                    ans = []
                    for y in range(threshold_image.shape[0]):
                        for x in range(threshold_image.shape[1]):
                            if threshold_image[y, x] > 0:
                                ans.append((y, x))
                    xx = [x for (y, x) in ans]
                    yy = [y for (y, x) in ans]
                    x = (max(xx) - min(xx)) / threshold_image.shape[1]
                    y = (max(yy) - min(yy)) / threshold_image.shape[0]
                    ellipse = cv2.minAreaRect(contours[i])
                    (center, axes, orientation) = ellipse
                    majoraxis_length = max(axes)
                    minoraxis_length = min(axes)
                    perimeter = cv2.arcLength(contours[i], True)
                    parametr_e = (lambda: (math.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)))()
                    parametr_c = (lambda: (4 * pi * area / perimeter ** 2))()
                    parametr_s = (lambda: ((area) /hull_area))()

                    wr.writerows([(parametr_s, parametr_e, parametr_c, hull_area, area, perimeter, square,
                               rh, x, y, R, std)])
                data_model = parametr_s, parametr_e, parametr_c, hull_area, area, perimeter, square, rh, x, y, R, std
                print(data_model)
    return data_model


if __name__=="__main__":
    dataset()
