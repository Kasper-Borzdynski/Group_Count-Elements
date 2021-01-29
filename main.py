import numpy as np
import cv2 as cv
import math
import argparse
import os
import json
import sys
import random




def emptycallback(value):
    pass
def niebieski_dziury():
    # color
    lower_blue = np.array([50, 50, 90])
    upper_blue = np.array([120, 256, 256])
    # brightness
    lower_bluebr = np.array([100, 230, 0])
    upper_bluebr = np.array([120, 256, 256])

    mask1 = cv.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv.inRange(hsv, lower_bluebr, upper_bluebr)

    mask = mask1 + mask2
    # segmentacja
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(close, cv.MORPH_OPEN, kernel)
    cnts, hierarchy = cv.findContours(opened.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    licz = 1
    for i in range(0, len(cnts)):
        area = cv.contourArea(cnts[i])
        if 1200 < area < 2200:
            (cx, cy), cr = cv.minEnclosingCircle(cnts[i])
            cr = int(cr)
            center = (int(cx), int(cy))
            cv.circle(img, center, cr, (0, 0, 255), 8)
            cv.circle(img, center, 5, (0, 0, 255), 5)
            #cv.putText(img, str(licz), center, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            licz = licz + 1
            xyb.append([int(cx), int(cy)])
def zolty_dziury():
    # color
    lower = np.array([20, 65, 150])
    upper = np.array([50, 256, 256])
    # brightness
    lower_br = np.array([156, 148, 255])
    upper_br = np.array([180, 256, 256])

    mask1 = cv.inRange(hsv, lower, upper)
    mask2 = cv.inRange(hsv, lower_br, upper_br)

    mask = mask1 + mask2
    # segmentacja
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(close, cv.MORPH_OPEN, kernel)
    cnts, hierarchy = cv.findContours(opened.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    licz = 1
    for i in range(0, len(cnts)):
        area = cv.contourArea(cnts[i])
        if 1200 < area < 2200:
            (cx, cy), cr = cv.minEnclosingCircle(cnts[i])
            cr = int(cr)
            center = (int(cx), int(cy))
            cv.circle(img, center, cr, (0, 0, 0), 8)
            cv.circle(img, center, 5, (0, 0, 255), 5)
            #cv.putText(img, str(licz), center, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            licz = licz + 1
            xyy.append([int(cx), int(cy)])
def czerwony_dziury():
    # color
    lower = np.array([0, 60, 115])
    upper = np.array([10, 255, 255])
    # brightness
    lower_br = np.array([170, 115, 100])
    upper_br = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower, upper)
    mask2 = cv.inRange(hsv, lower_br, upper_br)

    mask = mask1 + mask2
    # segmentacja
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    close = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    opened = cv.morphologyEx(close, cv.MORPH_OPEN, kernel)


    cnts, hierarchy = cv.findContours(opened.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    licz = 1
    for i in range(0, len(cnts)):
        area = cv.contourArea(cnts[i])
        if 1200 < area < 3000:
            (cx, cy), cr = cv.minEnclosingCircle(cnts[i])
            cr = int(cr)
            center = (int(cx), int(cy))
            cv.circle(img, center, cr, (0, 0, 0), 8)
            cv.circle(img, center, 5, (255, 0, 0), 5)
            #cv.putText(img, str(licz), center, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            licz = licz + 1
            xyr.append([int(cx),int(cy)])
            approx = cv.approxPolyDP(cnts[i], 0.1 * cv.arcLength(cnts[i], True), True)
            cv.drawContours(img, [approx], 0, (0, 0, 0), 10)
def bialy_dziury():
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    gray = cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=100.0, tileGridSize=(7, 7))
    cl1 = clahe.apply(gray)

    ret, thresh = cv.threshold(cl1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    open = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    dilate = cv.dilate(open, kernel, iterations=1)

    cnts, hierarchy = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in range(0, len(cnts)):
        area = cv.contourArea(cnts[i])
        M = cv.moments(cnts[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if 1200 < area < 2700:
            approx = cv.approxPolyDP(cnts[i], 0.1 * cv.arcLength(cnts[i], True), True)
            if 6 > len(approx) >= 4:
                cv.drawContours(img, [approx], 0, (0, 0, 0), 6)
                cv.circle(img, (cx, cy), 5, (0, 0, 0), 5)
                xyw.append([int(cx), int(cy)])
def szary_dziury():

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=50.0, tileGridSize=(3, 3))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    gray_blur = cv.GaussianBlur(final, (5, 5), 0)
    hsvg = cv.cvtColor(gray_blur, cv.COLOR_BGR2HSV)


    #hsvg = cv.resize(hsvg,None,0,fx=.2,fy=.2)
    #grey_grey = cv.resize(grey_grey,None,0,fx=.2,fy=.2)


    #lower = np.array([39, 0, 0])
    #upper = np.array([103, 58, 150])
    lower = np.array([35, 0, 10])
    upper = np.array([98, 255, 50])
    mask = cv.inRange(hsvg, lower, upper)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    close = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)

    cnts, hierarchy = cv.findContours(close.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    licz = 1
    for i in range(0, len(cnts)):
        area = cv.contourArea(cnts[i])
        if 1200 < area < 3000:
            (cx, cy), cr = cv.minEnclosingCircle(cnts[i])
            cr = int(cr)
            center = (int(cx), int(cy))
            cv.circle(img, center, cr, (0, 0, 255), 8)
            cv.circle(img, center, 5, (255, 0, 0), 5)
            #cv.putText(img, str(licz), center, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            licz = licz + 1
            xyg.append([int(cx), int(cy)])
def calc_dist(x1,y1,x2,y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist
def draw_mask(tab,mask,colo):
    onecolormask = np.zeros((h,w,3),np.uint8)
    for wixa in tab:
        cv.rectangle(mask,(wixa[0]-80,wixa[1]-80),(wixa[0]+80,wixa[1]+80),colo,-1)
        cv.rectangle(onecolormask,(wixa[0]-80,wixa[1]-80),(wixa[0]+80,wixa[1]+80),colo,-1)
    onecolormask = cv.cvtColor(onecolormask,cv.COLOR_BGR2GRAY)
    return onecolormask


pathimg = sys.argv[1]
pathjson = sys.argv[2]
pathoutput = sys.argv[3]

#pathimg = 'D:\\programowanie-python\\pycharmprojects\\projekt-SiSW'
#pathjson = 'D:\\programowanie-python\\pycharmprojects\\projekt-SiSW'
#pathoutput = 'D:\\programowanie-python\\pycharmprojects\\projekt-SiSW'


image = []
# r=root, d=directories, f = files
for r, d, f in os.walk(pathimg):
    for file in f:
        if '.jpg' in file:
            image.append(cv.imread(os.path.join(r, file), cv.IMREAD_COLOR))

iglo = 0
full_object_list = []
full_holes = []

for img in image:
    iglo = iglo + 1
    h,w,c = img.shape
    xyr = []
    xyb = []
    xyy = []
    xyw = []
    xyg = []
    object_list = []
    holes = []

    image_blur = cv.GaussianBlur(img,(5,5),0)
    zal = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image_blur, cv.COLOR_BGR2HSV)


    niebieski_dziury()
    zolty_dziury()
    czerwony_dziury()
    bialy_dziury()
    szary_dziury()

    object_mask = np.zeros((h,w,3),np.uint8)
    whity = (255,255,255)

    red = draw_mask(xyr,object_mask,whity)
    blue = draw_mask(xyb,object_mask,whity)
    yellow = draw_mask(xyy,object_mask,whity)
    white = draw_mask(xyw,object_mask,whity)
    gray = draw_mask(xyg,object_mask,whity)
    object_gray = cv.cvtColor(object_mask,cv.COLOR_BGR2GRAY)

    obiekty = 0
    cnts, hierarchy = cv.findContours(object_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        (rx, ry, rw, rh) = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        if area > 50000:
            r = 0
            b = 0
            w = 0
            g = 0
            y = 0

            count_holes = 0
            obiekty = obiekty + 1
            #cv.drawContours(img,[approx],0,(0,0,0),50)
            cv.rectangle(img,(rx,ry),(rx+rw,ry+rh),(11,128,200),4)
            cv.putText(img, str(obiekty), (rx,ry), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            redcnt, hier = cv.findContours(red,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            bluecnt, hierb = cv.findContours(blue,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            whitecnt, hierw = cv.findContours(white,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            graycnt, hierg = cv.findContours(gray,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            yellowcnt, hiery = cv.findContours(yellow,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

            for bb in xyb:
                if rx+75 < bb[0] < rx + rw - 75 and ry+75 < bb[1] < ry + rh-75:
                    count_holes = count_holes + 1
                    cv.circle(img, (bb[0], bb[1]), 20, (0, 0, 255), -1)
                    cv.putText(img, str(obiekty), (bb[0], bb[1]), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

            for rr in xyr:
                if rx+75 < rr[0] < rx + rw-75 and ry+75 < rr[1] < ry + rh-75:
                    count_holes = count_holes + 1
                    cv.circle(img, (rr[0], rr[1]), 20, (255, 0, 0), -1)
                    cv.putText(img, str(obiekty), (rr[0], rr[1]), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

            for ww in xyw:
                if rx+75 < ww[0] < rx + rw -75 and ry+75 < ww[1] < ry + rh-75:
                    count_holes = count_holes + 1
                    cv.circle(img, (ww[0], ww[1]), 20, (255, 255, 255), -1)
                    cv.putText(img, str(obiekty), (ww[0], ww[1]), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

            for gg in xyg:
                if rx + 75 < gg[0] < rx + rw - 75 and ry + 75 < gg[1] < ry + rh - 75:
                    count_holes = count_holes + 1
                    cv.circle(img, (gg[0], gg[1]), 20, (0, 0, 255), -1)
                    cv.putText(img, str(obiekty), (gg[0], gg[1]), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

            for yy in xyy:
                if rx+75 < yy[0] < rx + rw-75 and ry+75 < yy[1] < ry + rh-75:
                    count_holes = count_holes + 1
                    cv.circle(img, (yy[0], yy[1]), 20, (0, 0, 255), -1)
                    cv.putText(img, str(obiekty), (yy[0], yy[1]), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)

            for cc in redcnt:
                M = cv.moments(cc)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if rx < cx < rx+rw and ry < cy <= ry+rh:
                    approxx = cv.approxPolyDP(cc, 0.005 * cv.arcLength(cc, True), True)
                    cv.putText(img, str(obiekty), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
                    cv.drawContours(img,[approxx],0,(0,0,255),4)
                    r = r + 1

            for cc in bluecnt:
                M = cv.moments(cc)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if rx < cx < rx+rw and ry < cy <= ry+rh:
                    approxx = cv.approxPolyDP(cc, 0.005 * cv.arcLength(cc, True), True)
                    cv.drawContours(img,[approxx],0,(255,0,0),4)
                    cv.putText(img, str(obiekty), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
                    b = b + 1

            for cc in whitecnt:
                M = cv.moments(cc)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if rx < cx < rx+rw and ry < cy <= ry+rh:
                    approxx = cv.approxPolyDP(cc, 0.005 * cv.arcLength(cc, True), True)
                    cv.drawContours(img,[approxx],0,(255,255,255),4)
                    cv.putText(img, str(obiekty), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
                    w = w + 1

            for cc in graycnt:
                M = cv.moments(cc)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if rx < cx < rx+rw and ry < cy <= ry+rh:
                    approxx = cv.approxPolyDP(cc, 0.005 * cv.arcLength(cc, True), True)
                    cv.putText(img, str(obiekty), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
                    cv.drawContours(img,[approxx],0,(128,128,128),4)
                    g = g + 1

            for cc in yellowcnt:
                M = cv.moments(cc)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if rx < cx < rx+rw and ry < cy < ry+rh:
                    approxx = cv.approxPolyDP(cc, 0.005 * cv.arcLength(cc, True), True)
                    cv.putText(img, str(obiekty), (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
                    cv.drawContours(img,[approxx],0,(0,255,255),4)
                    y = y + 1

            object_list.append([r, b, w, g, y])
            holes.append(count_holes)

    #print(object_list)
    #print(holes)
    full_object_list.append(object_list)
    full_holes.append(holes)
    print(iglo)




    # img = cv.resize(img,None,0,fx=.2,fy=.2)
    # cv.imshow('loli', img)
    ch = cv.waitKey(0)
    cv.destroyAllWindows()

with open(pathjson + '/'  + 'public.json', 'r') as f:
    injson = json.load(f)

namearray = []
for key in injson:
    namearray.append(key)

full_real_holes = []
objcount = 0

for imgname in namearray:
    # wstaw kod
    real_hole = []
    #print('in obj:', str(objcount), sep=' ')
    #print(len(injson[imgname]), len(full_object_list[objcount]), sep='=')
    for k in range(0, len(injson[imgname])):

        match_full = False
        matches = 0
        not_found = []
        not_found_holes = []
        points = []
        for j in range(0, len(full_object_list[objcount])):
            point = 0
            dif = []
            dif.append(abs(full_object_list[objcount][j][0] - int(injson[imgname][k]["red"])))
            dif.append(abs(full_object_list[objcount][j][1] - int(injson[imgname][k]["blue"])))
            dif.append(abs(full_object_list[objcount][j][2] - int(injson[imgname][k]["white"])))
            dif.append(abs(full_object_list[objcount][j][3] - int(injson[imgname][k]["grey"])))
            dif.append(abs(full_object_list[objcount][j][4] - int(injson[imgname][k]["yellow"])))
            for dd in dif:
                if dd == 0:
                    point = point + 5
                elif dd == 1:
                    point = point + 3
                elif dd == 2:
                    point = point + 1
                elif dd == 3:
                    point = point + 0

            points.append(point)
        index = points.index(max(points))
        real_hole.append(full_holes[objcount][index])
    full_real_holes.append(real_hole)
    objcount = objcount + 1
numba = 0
outputjson = {}
for holes in full_real_holes:
    print(holes)
    outputjson[namearray[numba]] = holes
    numba = numba + 1

with open(pathoutput + '/'  + 'output.json', 'w') as json_file:
    json.dump(outputjson, json_file, indent=2)

