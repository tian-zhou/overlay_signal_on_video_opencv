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
from read_data import read_data

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

def draw_one_line(canvas, topleft, h, w, data, old_point, title, color):
    # topleft = (height, width)
    topleft = (int(topleft[0]), int(topleft[1]))
    N = len(data)

    # find critical points
    center_h = int(topleft[0] + 0.65 * h)
    start_w = int(topleft[1] + 0.01 * w)
    end_w = int(topleft[1] + 0.99 * w)

    # construct points (x,y)
    x_points = np.linspace(start_w, end_w, N).astype(int)

    # calculate scale
    if 0:
        scale = 40
        y_points = (center_h - scale * data).astype(int)
    elif 0:
        minx, maxx = np.amin(data), np.amax(data)
        y_points = (center_h - ((data-minx)/(maxx-minx)-0.5)*h).astype(int)
    else:
        if isinstance(old_point, int):
            scale = 40
            y_points = (center_h - scale * data).astype(int)
        else:
            minx, maxx = np.amin(data), np.amax(data)
            if minx == maxx: # for all 0, like kinect_audio
                y_points = np.append(old_point[1:], (center_h - 0)).astype(int)
            else:
                y_points = np.append(old_point[1:], (center_h - ((data[-1]-minx)/(maxx-minx)-0.5)* (0.7*h))).astype(int)

    # notice that small number should be large h value since y is pointing downwards
    points = np.array([x_points, y_points]).astype(np.int32).T

    # draw the line
    cv2.polylines(canvas, [points], isClosed=False, color=color, thickness=1)

    # put text to the right
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, org=(topleft[1], int(topleft[0]+0.2*h)), fontFace=font,
        fontScale=0.5, color=color, thickness=1)

    return canvas, y_points

def draw(h, w, data, old_points, colors):
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

    # add fake label for now
    label = np.ones((data.shape[0], 1))
##    data = np.concatenate((data, label), axis=1)

    # draw data
    N = data.shape[1]
    titles = {
    0: 'myo_orien_x',
    1: 'myo_orien_y',
    2: 'myo_acce_y',
    3: 'myo_gyro_y',
    4: 'kinect_face_y',
    5: 'kinect_face_z',
    6: 'kinect_lean_forward',
    7: 'kinect_audio',
    8: 'epoc_gyro_x',
    9: 'epoc_gyro_y',
##    10: 'turn_taking',
    }
    new_points = []
    for i in range(N):
        canvas, new_point = draw_one_line(canvas, (i*h/N, 0), int(float(h)/N),
            canvas_w, data[:,i], old_points[i], titles[i], colors[i])
        new_points.append(new_point)

    return canvas, new_points

def normalize_data(data):
    N, M = data.shape
    for i in range(M):
        if 0:
            mu = np.mean(data[:,i])
            std = np.std(data[:,i])
            data[:,i] = (data[:,i] - mu)/std
        else:
            minx = np.amin(data[:,i])
            maxx = np.amax(data[:,i])
            print "channel (%i) has minx (%.3f) maxx (%.3f)" % (i, minx, maxx)
            data[:,i] = (data[:,i] - minx)/(maxx-minx) # in range [0, 1]
            data[:,i] -= 0.5 # in range [-0.5, 0.5]
    return data

def main():
    # alpha
    alpha = 0.2
    colors = generate_good_color(11)

    # cap capture
    if 0:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture('../Take2/external_camera.MOV')

    if 0:
        # fake data
        data = np.random.rand(100,10) - 0.5
    else:
        data, abs_time = read_data(filename = '../Take2/raw_multimodal_data.txt',
            start=33685, end=35947)
        data = data[:,[15,16,19,22,40,41,128,130,146,147]]
        data = normalize_data(data)
        frame_idx = 0
        data_idx = 0


    # set init frame number
    videoProperty = {'posMillisec':0, 'frameCount':1, 'width':3, 'height':4,
                     'FPS':5, 'totalFrameCount':7}
    total_frame_count = cam.get(videoProperty['totalFrameCount']) # 67236
    FPS = cam.get(videoProperty['FPS']) # 30
    width = int(cam.get(videoProperty['width'])) # 1280
    height = int(cam.get(videoProperty['height'])) # 720

    # init video writer
    video_file = "output"
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter(video_file+'.avi',fourcc,30,(width,height))

    # set init frame count
    frameCount = 51200
    cam.set(videoProperty['frameCount'], frameCount)

    # start frame-wise drawing
    while True :
        # read one frame
        res, img_cam = cam.read()
        frame_idx += 1
        h,w,_ = img_cam.shape

        # data FPS is 20, and video FPS is 30
        if frame_idx % 3 == 0 or frame_idx % 3 == 2:
            # frame increase 3, data increase 2
            data_idx += 1
        vis_data = data[data_idx:data_idx+100, :]
        if data_idx+100 == len(data):
            print "data reaching end, break"
            break

        # create the canvas
        if frame_idx == 1:
            canvas, old_vis_data = draw(h, w, vis_data, [-1 for _ in range(data.shape[1]+1)], colors)
        else:
            canvas, old_vis_data = draw(h, w, vis_data, old_vis_data, colors)

        # add weighted and show
        output = np.zeros(img_cam.shape, np.uint8)
        if(res):
            # only add the left half
            canvas_w = int(0.2*w)
            cv2.addWeighted(img_cam, alpha, canvas, 1-alpha,  0, output)
            output[:, canvas_w:] = img_cam[:, canvas_w:]

            # write frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame_count = cam.get(videoProperty['frameCount'])
##            cv2.putText(output,'Frame: %i/%i' % (frame_count, total_frame_count),
##                (int(0.7 *w), int(0.95 * h)), font, 1, (0,0,255), 2)

            # show image
            cv2.imshow("output", output)
            video_out.write(output)
            key = cv2.waitKey(30)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    video_out.release()
    exit(-1)

if __name__ == '__main__':
    main()
