import os

import cv2

file_pattern = os.path.join('images', 'opencv')
if not os.path.exists('images'):
    os.makedirs('images')

# чтобы выйти из окна просмотра нажмите q

def process_img(i):
    image2 = cv2.imread(file_pattern + str(i) + '.png')
    start_point = (int(image2.shape[0] / 5), int(image2.shape[1] / 5))
    end_point = (int(image2.shape[0] / 5) + 100, int(image2.shape[1] / 5) + 100)
    cv2.rectangle(image2, start_point, end_point, color=(255, 0, 0), thickness=2)
    cv2.line(image2, start_point, end_point, color=(0, 255, 0), thickness=2)
    return image2


cv2.namedWindow("camera --- press q to stop")
camera = cv2.VideoCapture(0)
i = 0
images_arr = []

while camera.isOpened():
    return_value, image = camera.read()
    try:
        cv2.imwrite(file_pattern + str(i) + '.png', image)
        image2 = process_img(i)
        images_arr.append(image2)
        cv2.imshow("camera --- press q to stop", image2)
    except:
        camera.release()
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
camera.release()
cv2.destroyAllWindows()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# шейп почему-то перевернутый
# может кому-то с компом повезет больше
# а вообще интересный момент
output_size = image.shape[:2][::-1]
out = cv2.VideoWriter('output.mp4', fourcc, 23.0, output_size)
for frame in images_arr:
    out.write(frame)
out.release()

# на самом деле не знаю, нужны ли были эти файлы. Они часть более простого задания, так шо вот так вот
# удалю чтобы никто не заметил


for frame_dir in os.listdir('images'):
    os.remove(os.path.join('images', frame_dir))
