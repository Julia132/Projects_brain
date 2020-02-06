import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, \
    classification_report, confusion_matrix, accuracy_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn import metrics


data = []
labels = []

imagePaths = sorted(list(paths.list_images("part_start")))
ys = sorted(list(paths.list_images("part_finish_0")))
random.seed(100)
# random.shuffle(imagePaths)


for imagePath in imagePaths:
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = cv2.resize(image, (84, 84))
    data.append(image)

for y in ys:
    image_y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Image", image_y)
    # cv2.waitKey(0)
    # image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)
    image_y = image_y / 255.
    image_y = cv2.resize(image_y, (128, 128))
    labels.append(image_y)

#data = np.array(data, dtype="float") / 255.
# np.moveaxis(data,1,-1)

#labels = np.array(labels, dtype="float") / 255.
# np.moveaxis(labels,1,-1)
y = [1] * 98 + [0] * 98
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.25, random_state=100)
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.fit_transform(testY)

model = keras.models.Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3),  padding='same', input_shape=(84, 84, 1)))     #было 32
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))


# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(9, 9), strides=(1, 1), padding='same')) #было 256,  kernel_size=(11,11)
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))    #было 384
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))    #было 384
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))    #было 256
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPooling2D(pool_size=(2,  2), strides=(2, 2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(4096,)))
model.add(Activation('relu'))

# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))

# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(100))
model.add(Activation('relu'))

# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
INIT_LR = 0.01
opt = SGD(lr=INIT_LR)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
EPOCHS = 120

trainX = np.expand_dims(np.array(trainX), axis=3)
testX = np.expand_dims(np.array(testX), axis=3)


H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=2)
print("[INFO] evaluating network...")

predictions = model.predict_classes(testX, batch_size=32)
print(testY)
print(predictions)
matrix_metrics = classification_report(testY, predictions)
print(matrix_metrics)

precision, recall, fscore, support = precision_recall_fscore_support(testY, predictions)
_, specificity, _ = sensitivity_specificity_support(testY, predictions)
print('Accuracy', accuracy_score(testY, predictions))
print('binary precision value', precision[0])
print('binary recall value', recall[0])
print('binary fscore value', fscore[0])
print('binary specificity value', specificity[0])
#print(confusion_matrix(testY.round(), predictions))
#print(classification_report(testY, predictions))
dashList = [(5, 2), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2)]

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss", linestyle='--')
plt.plot(N, H.history["val_loss"], label="val_loss", linestyle='-.')
plt.plot(N, H.history["acc"], label="train_acc", linestyle= ':')
plt.plot(N, H.history["val_acc"], label="val_acc",  linestyle='-')
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("Training Loss and Accuracy.pdf")
plt.show()


plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(testY, predictions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('AlexNet', roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Correctly defined neoplasms')
plt.ylabel('Fallaciously defined neoplasms')
plt.legend(loc=0, fontsize='small')
plt.title("ROC - curve")
plt.show()

plt.figure(figsize=(8, 8))
recall = metrics.recall_score(testY, predictions, average=None)
specificity = recall[0]
precision, recall, thresholds = metrics.precision_recall_curve(testY, predictions)
plt.plot(recall, precision)
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Curve dependent Precision и Recall of threshold")
plt.legend(loc='best')
plt.show()

x, y = [], []

for threshold in np.arange(0.0, 1.0, 0.01).tolist():
    y_pred = np.where(predictions >= threshold, 1, 0)
    recall = metrics.recall_score(testY, y_pred, average=None)
    x.append(recall[0])
    y.append(recall[1])
plt.figure()
plt.plot(x, y)
plt.xlabel("Correctly defined with neoplasms")
plt.ylabel("Correctly defined without neoplasms")
plt.show()