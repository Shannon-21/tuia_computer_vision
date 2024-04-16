import cv2
from datetime import datetime
import time
import numpy as np

# Function for motion detection
def motion_detection(video):
    initialState = None
    motionTrackList = [None, None]
    motionTime = []

    while True:
        check, cur_frame = video.read()
        var_motion = 0

        if not check:
            break  # If there are no more frames to read, break the loop

        gray_image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Adjust kernel size

        if initialState is None:
            initialState = gray_frame
            continue

        differ_frame = cv2.absdiff(initialState, gray_frame)
        thresh_frame = cv2.threshold(differ_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cont, _ = cv2.findContours(thresh_frame.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cur in cont:
            if cv2.contourArea(cur) < 500:  # Adjust contour area threshold
                continue

            var_motion = 1
            (cur_x, cur_y, cur_w, cur_h) = cv2.boundingRect(cur)

            cv2.rectangle(cur_frame, (cur_x, cur_y), (cur_x + cur_w, cur_y + cur_h), (0, 255, 0), 3)

        motionTrackList.append(var_motion)
        motionTrackList = motionTrackList[-2:]

        if motionTrackList[-1] == 1 and motionTrackList[-2] == 0:
            motionTime.append(datetime.now())

        if motionTrackList[-1] == 0 and motionTrackList[-2] == 1:
            motionTime.append(datetime.now())

        cv2.imshow("Video", cur_frame)
        time.sleep(0.01)

        wait_key = cv2.waitKey(1)
        if wait_key == ord('m'):
            if var_motion == 1:
                motionTime.append(datetime.now())
            break

# Function for dense optical flow
def dense_optical_flow(video):
    ret, frame1 = video.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = video.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Dense Optical Flow', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        prvs = next

        time.sleep(0.01)

# Function for sparse optical flow
def sparse_optical_flow(video):
    feature_params = dict(maxCorners = 1000, qualityLevel = 0.01, minDistance = 1, blockSize = 7)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = (0, 255, 0)
    ret, first_frame = video.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    mask = np.zeros_like(first_frame)

    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        good_old = prev[status == 1]
        good_new = next[status == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 3, color, -1)

        output = cv2.add(frame, mask)
        prev_gray = gray.copy()
        prev = good_new.reshape(-1, 1, 2)
        cv2.imshow("sparse optical flow", output)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Abre el video
video = cv2.VideoCapture('Unidad1\Videos\security_camera.mp4')

# Aplica detección de movimiento
motion_detection(video)

# Reinicia el video
video.release()
video = cv2.VideoCapture('Unidad1\Videos\security_camera.mp4')

# Aplica flujo óptico denso
dense_optical_flow(video)

# Reinicia el video
video.release()
video = cv2.VideoCapture('Unidad1\Videos\security_camera.mp4')

# Aplica flujo óptico disperso
sparse_optical_flow(video)

# Libera los recursos y cierra las ventanas
video.release()
cv2.destroyAllWindows()
