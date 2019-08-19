import cv2  # графический модуль opencv
import numpy as np  # модуль операций над матрицами

#пока этот модуль не особо нужен

#from brain_main_git import mask5
def characteristic(mask5):
    image = mask5
    image = cv2.resize(image, (512, 512))
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        hull = cv2.convexHull(contours[i])
        hull_area = cv2.contourArea(hull)
        _, radius = cv2.minEnclosingCircle(contours[i])
        radius = int(radius)
        if 0 != radius:
            square = area / (image.shape[0]*image.shape[1])
            rh = radius / image.shape[0]
            ans = []
            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    if image[y, x] > 0:
                        ans.append((y, x))
            xx = [x for (y, x) in ans]
            yy = [y for (y, x) in ans]
            x = (max(xx) - min(xx)) / image.shape[1]
            y = (max(yy) - min(yy)) / image.shape[0]
            ellipse = cv2.minAreaRect(contours[i])
            (center, axes, orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            perimeter = cv2.arcLength(contours[i], True)
            from math import pi  # для константы pi
            import math
            parametr_e = (lambda: (math.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)))()
            parametr_c = (lambda: (4 * pi * area / perimeter ** 2))()
            parametr_s = (lambda: ((area) /hull_area))()
            metrics = parametr_s, parametr_e, parametr_c, hull_area, area, perimeter, square, rh, x, y
            print(metrics)
    return metrics, parametr_s
metrics = characteristic(mask5)
if __name__=="__main__":
    characteristic(mask5)
