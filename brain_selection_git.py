import numpy as np  # модуль операций над матрицами
from os import listdir  # список файлов в папке
from os.path import isfile, join  # методы проверки файла и соединения с файлом
from sklearn.metrics import classification_report  # для вывода метрик
from sklearn.metrics import precision_recall_fscore_support   # для вывода метрик
from sklearn.metrics import accuracy_score
from imblearn.metrics import sensitivity_specificity_support

def selection():
    mypath_out = 'C:/Users/inet/Desktop/part_finish_0'  #больные и здоровые в ручной обработке
    mypath_no = 'C:/Users/inet/Desktop/no pathologies'  #здоровые по мнению кода
    mypath_start_no = 'C:/Users/inet/Desktop/not'   #здоровые в серых тонах

    onlyfiles_out = [f for f in listdir(mypath_out) if isfile(join(mypath_out, f))]
    onlyfiles_no = [f for f in listdir(mypath_no) if isfile(join(mypath_no, f))]

    print('Число снимков без патологий', len(onlyfiles_no))
    print('Число снимков c патологиями', len(onlyfiles_out) - len(onlyfiles_no))
    dir_no = set(listdir(mypath_no))
    dir_late = set(listdir(mypath_start_no))

    y_true = [f not in dir_no for f in onlyfiles_out]
    y_pred = [f not in dir_late for f in onlyfiles_out]

    list_precision = []
    list_recall = []
    list_fscore = []
    list_specificity = []
    list_accuracy = []

    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    _, specificity, _ = sensitivity_specificity_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    list_precision.append(precision)
    list_recall.append(recall)
    list_fscore.append(fscore)
    list_specificity.append(specificity)
    list_accuracy.append(accuracy)

    Avg_precision = np.mean(list_precision)
    Avg_recall = np.mean(list_recall)
    Avg_fscore = np.mean(list_fscore)
    Avg_specificity = np.mean(list_specificity)
    Agv_accuracy = np.mean(list_accuracy)
    print('weighted precision value', Avg_precision)
    print('weighted recall value', Avg_recall)
    print('weighted fscore value', Avg_fscore)
    print('weighted specificity value', Avg_specificity)
    print('weighted accuracy value', Agv_accuracy)
    return Agv_accuracy, Avg_precision, Avg_recall, Avg_fscore, Avg_specificity
# Agv_accuracy = selection()


if __name__=="__main__":
    selection()