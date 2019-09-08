import cv2

video_object = cv2.VideoCapture('a.mp4')
fps = video_object.get(cv2.CAP_PROP_FPS)

print(fps)
