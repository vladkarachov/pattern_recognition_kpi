import numpy as np
import cv2

import time

vid = cv2.VideoCapture('input.mp4')
file = open('encoded.npy', 'wb')

BS = 8 #розмір макроблоку
Step = 4 #залежить пошукова область(2*Step-1)
S2=1
sum_of_err=0


def findbias(first, second):
    bias = np.zeros((first.shape[0]//BS, first.shape[1]//BS, 2))
    onethread(bias, 0, first.shape[0]//BS, first, second, first.shape[1]//BS, sum_of_err)
    return bias
def onethread(bias, fro, to, first, second, width, sum_of_err):
    for i in range(fro, to):
        for j in range(width):
            S=Step
            imin = 0
            jmin = 0
            while(S > 0):
                im, jm, norm= findmindif(first, second, S, i*BS+imin, j*BS+jmin)
                imin += im
                jmin += jm
                S = S // 2
            sum_of_err+=norm
            bias[i, j, 0] = imin
            bias[i, j, 1] = jmin

def findmindif(first, second, S, x, y):
    block = first[x:x+BS , y:y+BS]
    mind = mad(block, second[x:x+BS , y:y+BS])
    imin, jmin = 0,0
    positions  =  [(i,j) for i in range(-S,S+1,8) for j in range(-S,S+1,8) if (i,j)!=(0,0)]
#    positions +=  [(i,j) for i in range(-S2,S2+1) for j in range(-S2,S2+1) if (i,j)!=(0,0)]
    for pos in positions:
        i,j=pos
        if x + i >= 0 and x + i + BS <= second.shape[0] \
                and y + j >= 0 and y + j + BS <= second.shape[1]:
            norm = mad(block, second[x+i:x+i+BS,y+j:y+j+BS])
            if mind > norm:
                mind = norm
                imin,jmin = i,j
    return imin, jmin, mind


def buildFrame(frame, bias):
    newframe = frame.copy()
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            block = frame[i * BS:(i + 1) * BS, j * BS:(j + 1) * BS, :]
            newframe[int(i * BS + bias[i][j][0]):int((i + 1) * BS + bias[i][j][0]),
            int(j * BS + bias[i][j][1]):int((j + 1) * BS + bias[i][j][1]), :] = block
    return newframe


# Mean Absolute Difference
def mad(a, b):
    return 1 / a.shape[0] / a.shape[1] * np.sum(np.abs(a - b))
    # return 1/BS**2*np.sum(np.sum(np.abs(a - b), axis=1),axis=0)

difference = 0
count = 0

start_time = time.time()
while vid.isOpened():
    ret, frame1 = vid.read()
    if ret:
        ret, frame2 = vid.read()
        if ret:
            np.save(file, frame1)

            first = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            second = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            bias = findbias(second, first)
            newframe = buildFrame(frame1, bias)
            difference += mad(frame2, newframe)
            count += 1
            np.save(file, bias)

        else:
            np.save(file, frame1)
            break
    else:
        break

times = round(time.time() - start_time, 0)
print(f"time of encoding: {times // 60} minutes {times % 60} seconds")
print(f"mean difference between original and generated frames: {round(difference/count, 4)}")


vid.release()
file.close()
cv2.destroyAllWindows()
