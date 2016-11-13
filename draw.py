"""
-------------------------------------------------------------------------------
Name:        draw signals on the video
Author:      Tian Zhou
Email:       zhou338 [at] purdue [dot] edu
Created:     11/12/2016
Copyright:   (c) Tian Zhou 2016
-------------------------------------------------------------------------------
"""

#!/usr/bin/env python
import cv2
import numpy as np

def generate_good_color(N):
    # generrate N distinct colors
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.rand()
    colors = []
    for i in range(N):
        h += golden_ratio_conjugate
        h %= 1
        hsv = (255 * np.array([h, 0.5, 0.95])).astype(np.uint8).reshape(1,1,3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(bgr.reshape(-1).astype(int))
    return colors

def draw_one_line(canvas, topleft, h, w, data, title, color):
    # topleft = (height, width)
    topleft = (int(topleft[0]), int(topleft[1]))
    N = len(data)

    # find critical points
    center_h = int(topleft[0] + 0.5 * h)
    start_w = int(topleft[1] + 0.01 * w)
    end_w = int(topleft[1] + 0.99 * w)

    # construct points (x,y)
    x_points = np.linspace(start_w, end_w, N).astype(int)
    scale = 30
    y_points = (center_h + scale * data).astype(int)
    points = np.array([x_points, y_points]).astype(np.int32).T

    # draw the line
    cv2.polylines(canvas, [points], isClosed=False, color=color, thickness=1)

    # put text to the right
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, org=(topleft[1], int(topleft[0]+0.2*h)), fontFace=font,
        fontScale=0.5, color=color, thickness=1)

    return canvas

def draw(h, w, data, colors):
    # init canvas
    canvas = np.zeros((h,w,3), np.uint8)
    canvas_w = int(0.2*w)

    # create transparent background
    if 0:
        canvas[int(0.0*h):int(0.33*h), :canvas_w, 0] = 255
        canvas[int(0.33*h):int(0.67*h), :canvas_w, 1] = 255
        canvas[int(0.67*h):int(1.0*h), :canvas_w, 2] = 255
    else:
        canvas[int(0.0*h):int(1.0*h), :canvas_w, :] = 0

    # draw data
    N = data.shape[1]
    titles = {
    0: 'myo_orienx',
    1: 'myo_orieny',
    2: 'myo_orienz',
    3: 'kinect_lean',
    4: 'kinect_sound',
    5: 'kinect_music',
    6: 'kinect_body',
    7: 'epoc_gyrox',
    8: 'epoc_gyroy',
    9: 'turn_taking',
    }
    for i in range(N):
        canvas = draw_one_line(canvas, (i*h/N, 0), int(float(h)/N), canvas_w, data[:,i], titles[i], colors[i])

    return canvas

def main():
    # alpha
    alpha = 0.5
    colors = generate_good_color(10)

    # cap capture
    cam = cv2.VideoCapture(0)
    data = np.random.rand(100,10) - 0.5
    while True :
        # read one frame
        res, img_cam = cam.read()
        h,w,_ = img_cam.shape

        # create the canvas
        data = np.concatenate((data[1:, :], np.random.rand(1,10)-0.5), axis=0)
        canvas = draw(h, w, data, colors)

        # add weighted and show
        output = np.zeros(img_cam.shape, np.uint8)
        if(res):
            # only add the left half
            canvas_w = int(0.2*w)
            cv2.addWeighted(img_cam, alpha, canvas, 1-alpha,  0, output)
            output[:, canvas_w:] = img_cam[:, canvas_w:]

            cv2.imshow("output", output)
            key = cv2.waitKey(33)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(-1)

if __name__ == '__main__':
    main()
