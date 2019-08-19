import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from imblearn.metrics import sensitivity_specificity_support

def csv_model():
    dataset = pd.read_csv("C:/Users/inet/Documents/GitHub/Projects_brain/data_set_2.csv", sep=',', encoding='latin1',
                          dayfirst=True,
                          index_col=None, header=None)

    y = [1] * 98 + [0] * 98

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=0)

    print("Оригинальные размеры данных: ", X_train.shape, X_test.shape)

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
    plt.show()

    y_pred = classifier.predict(X_test)
    matrix_metrics = classification_report(y_test, y_pred)
    print(matrix_metrics)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    _, specificity, _ = sensitivity_specificity_support(y_test, y_pred)

    print('Accuracy',  accuracy_score(y_test, y_pred))
    print('binary precision value', precision[0])
    print('binary recall value', recall[0])
    print('binary fscore value', fscore[0])
    print('binary specificity value', specificity[0])
    y = pd.DataFrame(y_pred)
    y.to_csv('out.csv', index=False, header=False)


if __name__=="__main__":
    csv_model()


