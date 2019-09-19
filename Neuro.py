import keras
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from imblearn.metrics import sensitivity_specificity_support
import cv2
import os
EMBED_DIM = 100
np.random.seed(1000)

data = []
labels = []

# берём пути к изображениям и рандомно перемешиваем
imagePaths = sorted(list(paths.list_images("C:/Users/inet/Desktop/part_start")))
ys = sorted(list(paths.list_images("C:/Users/inet/Desktop/part_finish_0")))
random.seed(200)
# random.shuffle(imagePaths)

# цикл по изображениям
for imagePath in imagePaths:
    # загружаем изображение, меняем размер на 32x32 пикселей (без учёта
    # соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
    # добавляем в список
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = cv2.resize(image, (72, 72))
    data.append(image)
    # извлекаем метку класса из пути к изображению и обновляем
    # список меток
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
y = [1] * 98 + [0] * 102
trainX, testX, trainY, testY = train_test_split(data, y, test_size=0.25, random_state=100)
# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.fit_transform(testY)

model = keras.models.Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3),  padding='same', input_shape=(72, 72, 1)))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

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
# trainY = np.expand_dims(np.array(trainY), axis=3)
testX = np.expand_dims(np.array(testX), axis=3)
# testY = np.expand_dims(np.array(testY), axis=3)

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

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()


