import os
import cv2

img_path = 'video/poker_3E/ours/'

list = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        list.append(file)
list.sort()
print(list)

#--------------------批量修改图像名字适应video的顺序--------------------------------
# for i in range(1, len(list)):
#     if len(list[i-1]) == 11:
#         before = './results/model_3E_test/dynamic_dataset_noGT/' + list[i-1]
#         after = './results/model_3E_test/dynamic_dataset_noGT/' + '0' + list[i-1]
#         os.rename(before, after)

video = cv2.VideoWriter('video/poker_3E/ours/ours_stop_poker_video.avi',cv2.VideoWriter_fourcc(*'MJPG'),5,(1890,1050))#(1890,1050) or (1280,720)
for i in range(16):
    img = cv2.imread(img_path + list[i])
    video.write(img)
video.release()