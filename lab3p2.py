import os

import cv2
import numpy as np


def add_samples(images, label, descriptor, matcher, disc0, MIN_MATCH_COUNT):
    train_data = []
    y = []
    for im in images:
        k, d = descriptor.detectAndCompute(im, None)
        try:
            matches = matcher.knnMatch(disc0, d, k=2)
            goodMatches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:  # vary
                    goodMatches.append(m)
            if len(goodMatches) < MIN_MATCH_COUNT:
                raise Exception("min_match_count")
            #вырезаем из кадра дескрпторы и добавляем их в семпл
            dest_matches = np.zeros(disc0.shape)
            for m in goodMatches:
                dest_matches[m.queryIdx, :] = d[m.trainIdx, :]
            train_data.append(dest_matches.ravel() / 256)
            y.append(label)
        except:
            print("min_match_count")
    return train_data, y


# loading im
file_pattern = ''
file_pattern = os.path.join('data_lb2', 'im', 'photo')
im1_path = file_pattern + 'main.jpg'
im2_path = file_pattern + '93.jpg'

# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
# img2 = cv2.imread(im2_path, cv2.IMREAD_COLOR)
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img1 = cv2.imread('0.jpg', cv2.IMREAD_COLOR)

if img1 is None:
    print('Could not open or find the images!')
    exit(0)

scale_percent = 40  # percent of original size

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# akaze
#akaze = cv2.AKAZE_create(threshold=0.01)
akaze = cv2.AKAZE_create(threshold=0.001)
key0, disc0 = akaze.detectAndCompute(img1, None)
print("descriptor sizes: ")
print(disc0.shape)

images = []
imagesn = []
img_positive = 99
img_negative = 100

for i in range(img_positive):
    img = cv2.imread(file_pattern + f'{i}.jpg', cv2.IMREAD_COLOR)
    if os.path.exists(file_pattern + f'{i}.jpg'):
        # я удалаял файлы руками(
        images.append(img)
for i in range(1, img_negative):
    img = cv2.imread(f'img/{i}.jpg', cv2.IMREAD_COLOR)
    if os.path.exists(f'img/{i}.jpg'):
        imagesn.append(img)

# выбираем только хорошие дескрипторы, а то размерность всего етого дела будет нереальная
bf = cv2.BFMatcher()

# считаем для всех изображений дескриптор аказе
# меняем силу дескриптора чтобы не получать ошибки

train_data = []
y = []
MIN_MATCH_COUNT = 5
tp, yp = add_samples(images, 1, akaze, bf, disc0, MIN_MATCH_COUNT)
train_data += tp
y += yp
tn, yn = add_samples(imagesn, 0, akaze, bf, disc0, MIN_MATCH_COUNT)
train_data += tn
y += yn

train_data = np.array(train_data)
y = np.array(y)
print("Train shape: ")
print(train_data.shape)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=400)
scores = cross_val_score(clf, train_data, y, cv=5)
print(scores.mean())
