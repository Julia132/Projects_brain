import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, \
    accuracy_score, confusion_matrix
from imblearn.metrics import sensitivity_specificity_support


def csv_model():

    """""

    This function classification geometric features from .csv, analysis of job program

    """""

    dataset = pd.read_csv("data_set_geometric_features.csv", sep=',', encoding='latin1',
                          dayfirst=True,
                          index_col=None, header=None)

    y = [1] * 96 + [0] * 96

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=0)

    print("Original data_set sizes: ", X_train.shape, X_test.shape)

    sc = StandardScaler(copy=True, with_mean=True, with_std=True)

    X_train = sc.fit_transform(X_train.astype(float))
    X_test = sc.transform(X_test.astype(float))

    classifier = RandomForestClassifier(n_estimators=196, criterion='gini', random_state=0, max_depth=10,
                                        min_samples_split=14, max_features=6, )
    classifier.fit(X_train, y_train)
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
                 axis=0)

    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    for f in range(dataset.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(dataset.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(dataset.shape[1]), indices)
    plt.xlim([-1, dataset.shape[1]])


    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    _, specificity, _ = sensitivity_specificity_support(y_test, y_pred)

    print('Accuracy: ',  accuracy_score(y_test, y_pred))
    print('Precision value: ', precision[0])
    print('Recall value: ', recall[0])
    print('F-score value: ', fscore[0])
    print('Specificity value: ', specificity[0])

    result = pd.DataFrame(y_pred)
    result.to_csv('result_RandomForest.csv', index=False, header=False)


if __name__=="__main__":
    csv_model()


