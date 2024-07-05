import numpy as np
import cv2
import sys
import os

rect = (0, 0, 0, 0)
start_point = False
end_point = False
img_counter = 0


BACKGROUND = cv2.imread(f"./images/beach.jpg")

if len(sys.argv) < 2:
    print("No output directory specified, quitting")
    quit()

output_dir = sys.argv[1]
try:
    os.mkdir(output_dir)
except FileExistsError:
    print("There is already a directory with specified name, quitting")
    quit()


# function for mouse callback
def on_mouse(event, x, y, flags, params):
    global rect, start_point, end_point

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if start_point == True and end_point == True:
            start_point = False
            end_point = False
            rect = (0, 0, 0, 0)

        if start_point == False:
            rect = (x, y, 0, 0)
            start_point = True
        elif end_point == False:
            rect = (rect[0], rect[1], x, y)
            end_point = True


# cap = cv2.VideoCapture("YourVideoFile.mp4")

# capturing the camera feed, '0' denotes the first camera connected to the computer
cap = cv2.VideoCapture(0)
waitTime = 50

# Reading the first frame
(grabbed, frame) = cap.read()

while cap.isOpened():

    (grabbed, frame) = cap.read()

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_mouse)

    # drawing rectangle
    if start_point == True and end_point == True:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)

    if not grabbed:
        break
    cv2.imshow("frame", frame)

    key = cv2.waitKey(waitTime)
    if key == 27:
        # esc pressed
        break

    elif key % 256 == 32:
        # SPACE pressed
        alpha = 1  # Transparency factor.

        img_name = f"opencv_frame_{img_counter}.png"

        img = frame

        mask = np.zeros(img.shape[:2], np.uint8)
        resized_background = cv2.resize(BACKGROUND, (img.shape[1], img.shape[0]))

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        w = abs(rect[0] - rect[2] + 10)
        h = abs(rect[1] - rect[3] + 10)
        rect2 = (rect[0] + 10, rect[1] + 10, w, h)

        cv2.grabCut(img, mask, rect2, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img = img * mask2[:, :, np.newaxis]

        img = np.where(img == 0, resized_background, img)

        cv2.imwrite(f"{output_dir}/{img_name}", img)
        cv2.imwrite(f"{output_dir}/og_{img_name}", frame)
        print(f"{img_name} written!")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
