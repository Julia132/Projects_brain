import cv2  # графический модуль opencv
import numpy as np  # модуль операций над матрицами
from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом
from sklearn.metrics import classification_report  # для вывода метрик
from sklearn.metrics import precision_recall_fscore_support   # для вывода метрик
from sklearn.metrics import accuracy_score
from math import pi  # для константы pi
from imblearn.metrics import sensitivity_specificity_support


def main():
    mypath_in = 'C:/Users/inet/Desktop/part_start'
    mypath_out = 'C:/Users/inet/Desktop/part_finish'
    mypath_late = 'C:/Users/inet/Desktop/part_start_late'
    mypath_no = 'C:/Users/inet/Desktop/no pathologies'
    onlyfiles = [f for f in listdir(mypath_in) if isfile(join(mypath_in, f))]
    onlyfiles_out = [f for f in listdir(mypath_out) if isfile(join(mypath_out, f))]
    onlyfiles_no = [f for f in listdir(mypath_no) if isfile(join(mypath_no, f))]
    img = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        img[n] = cv2.imread(join(mypath_in, onlyfiles[n]))
        gray = cv2.cvtColor(img[n], cv2.COLOR_BGR2GRAY)

        def standardized(gray):
            newImg = cv2.resize(gray, (512, 512))
            R = np.mean(newImg)
            std = np.std(newImg)
            standardized_images_out = ((newImg - R) / std) * 40 + 127
            blur = cv2.blur(standardized_images_out, (7, 7))
            return blur

        blur = standardized(gray)

        def morph(blur):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, 1)
            thresh = cv2.adaptiveThreshold(opening.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            dilation = cv2.dilate(thresh, kernel, 1)
            erode = cv2.erode(dilation, kernel, 3)
            closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel, 5)
            return closing

        closing = morph(blur)

        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(closing.shape, np.uint8)
        def selection(contours):
            hull = []
            for i in range(0, len(contours)):
                hull.append(cv2.convexHull(contours[i], False))
                cv2.drawContours(drawing, contours, i, 255, 1, 51, hierarchy)
                closing = cv2.drawContours(drawing, hull, i, 100, 1, 8)
            mask = np.zeros(closing.shape, np.uint8)
            for i in range(0, len(contours)):
                area = cv2.contourArea(contours[i])
                hull = cv2.convexHull(contours[i])
                hull_area = cv2.contourArea(hull)
                _, radius = cv2.minEnclosingCircle(contours[i])
                radius = int(radius)
                if 300 < area < 10000 and 10 < radius < 100:

                        ellipse = cv2.fitEllipse(contours[i])
                        (center, axes, orientation) = ellipse
                        majoraxis_length = max(axes)
                        minoraxis_length = min(axes)
                        eccentricity = (np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2))
                        perimeter = cv2.arcLength(contours[i], True)
                        circularity = 4 * pi * (area / (perimeter * perimeter))
                        solidity = float(area) / hull_area
                        if 0.2 < circularity < 1.2 and 0.1 < eccentricity < 0.9 and solidity > 0.7:
                            cv2.drawContours(mask, contours, i, 255, -1)
            if int(np.mean(mask)) >= 1:
                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                Max = 0
                pos = 0
                mask2 = np.zeros(mask.shape, np.uint8)
                for idx in range(0, len(contours)):
                    area = cv2.contourArea(contours[idx])
                    if area > Max:
                        Max = area
                        pos = idx
                Contour = np.array(contours[pos])
                cv2.drawContours(mask2, [Contour], 0, 255, -1)
                mask3 = cv2.medianBlur(mask2, 9)
                mask4 = cv2.medianBlur(mask3, 9)
                mask5 = cv2.medianBlur(mask4, 9)
                cv2.imwrite(join(mypath_late, onlyfiles[n]), mask5)
            if int(np.mean(mask)) == 0:
                cv2.imwrite(join(mypath_no, onlyfiles[n]), mask)
        selection(contours)
    list_precision = []
    list_recall = []
    list_fscore = []
    list_specificity = []
    list_accuracy = []
    start_img = np.empty(len(onlyfiles_out), dtype=object)
    final_img = np.empty(len(onlyfiles_out), dtype=object)
    onlyfiles_late = [f for f in listdir(mypath_late) if isfile(join(mypath_late, f))]
    onlyfile_out = [f for f in listdir(mypath_out) if isfile(join(mypath_out, f))]
    for n in range(0, len(onlyfiles_late)):
        start_img[n] = cv2.imread(join(mypath_late, onlyfiles_late[n]), cv2.IMREAD_GRAYSCALE) #ошибка в количестве файлов
        final_img[n] = cv2.imread(join(mypath_out, onlyfiles_late[n]), cv2.IMREAD_GRAYSCALE)
        gray_s = cv2.resize(start_img[n], (512, 512))
        gray_f = cv2.resize(final_img[n], (512, 512))
        _, start = cv2.threshold(gray_s, 127, 255, cv2.THRESH_BINARY)
        _, final = cv2.threshold(gray_f, 127, 255, cv2.THRESH_BINARY)
        y_pred = start.ravel()
        y_true = final.ravel()
        print(onlyfiles_out[n], classification_report(y_true, y_pred))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true == 255, y_pred == 255, pos_label=True, average="binary")
        _, specificity, _ = sensitivity_specificity_support(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        list_precision.append(precision)
        list_recall.append(recall)
        list_fscore.append(fscore)
        list_specificity.append(specificity[0])
        list_accuracy.append(accuracy)

    Avg_precision = np.mean(list_precision)
    Avg_recall = np.mean(list_recall)
    Avg_fscore = np.mean(list_fscore)
    Avg_specificity = np.mean(list_specificity)
    Agv_accuracy = np.mean(list_accuracy)
    print('binary precision value', Avg_precision)
    print('binary recall value', Avg_recall)
    print('binary fscore value', Avg_fscore)
    print('binary specificity value', Avg_specificity)
    print('binary accuracy value', Agv_accuracy)
    return contours
contours = main()

