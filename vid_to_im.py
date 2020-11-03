import os

import cv2


# блин лень делать 100 фоток((
# video = cv2.VideoCapture('data_lb2/vid.mp4')
# i = 0
# file_pattern = os.path.join('data_lb2', 'im', 'photo')
# while video.isOpened():
#     return_value, image = video.read()
#     try:
#         cv2.imwrite(file_pattern + str(i) + '.jpg', image)
#     except:
#         video.release()
#         break
#     i += 1
# print('complete with ' + str(i) + ' images')


def createimages(path):
    video = cv2.VideoCapture(path)
    frames = []
    i=0
    while video.isOpened():
        return_value, image = video.read()
        if(return_value):
            frames.append(image)
        else:
            video.release()
            break
        i+=1
        if i%20==0 :print('\n'+str(i))
    print('frames reading comlete')
    return frames


def writeimages(frames):
    i = 0
    file_pattern = os.path.join('data_lb2', 'im', 'photo')
    for frame in frames:
        cv2.imwrite(file_pattern + str(i) + '.jpg', frame)
        i += 1
    print('complete with ' + str(i) + ' images')


frames = createimages('data_lb2/vid.mp4') + createimages('data_lb2/vid2.mp4')
step = len(frames)//95
frames = frames[0::step]
writeimages(frames)
