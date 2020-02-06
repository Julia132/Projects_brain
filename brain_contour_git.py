import cv2
import numpy as np

from os import listdir
from os.path import isfile, join


def contour():


    mypath_late = 'part_start'
    mypath_contour = 'contour'
    mypath_out = 'part_finish'

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


edges = contour()