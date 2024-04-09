# tensorflow==1.13.1 or tensorflow-gpu==1.13.1 if CUDA support available
# keras==2.2.4
# numpy==1.16.1
# imageai==2.1.5
# opencv==4.3.0.38

import logging
import os
import os.path
import time
from time import sleep
from itertools import groupby

import cv2
import numpy as np
import requests  # to use HTTP POST request
from imageai.Detection.Custom import CustomObjectDetection

# ----- Necessary Variables -----
model_path = "detection_model-ex-032--loss-0001.977.h5"
json_path = "detection_config2.json"

input_image_path = "C:/imgProc/"  # folder path
output_image_path = "C:/imgProc/detected.jpg"
minimum_percentage_probability = 80
nms_threshold = 0.4  # default nms is 0.4 - change this if an object founded two or more times

# HTTP POST request variables
url = ''
mac = ''
temp = 0
latitude = 0
longitude = 0
json_data = ""
# -------------------------------


def get_stock_info(filename):
    global coca_cola_kutu_counter
    global coca_cola_cam_counter
    global coca_cola_sise_counter
    global fanta_kutu_counter
    global fanta_cam_counter
    global fanta_sise_counter
    global sprite_kutu_counter
    global sprite_cam_counter
    global sprite_sise_counter

    # Open image in BGR(Blue-Green-Red) color space
    img = cv2.imread(input_image_path + filename)

    if img is None:  # the imread() function returns a NoneType object on error.
        logging.error('The image is empty.')
        print('The image is empty. Please check if image exits.')
    else:
        # Get predictions from the model
        detections = detector.detectObjectsFromImage(input_image=input_image_path + filename,
                                                     output_image_path=output_image_path,
                                                     minimum_percentage_probability=minimum_percentage_probability,
                                                     nms_treshold=nms_threshold)

        os.remove(input_image_path + filename)  # delete image from folder

        # Convert image from BGR(Blue-Green-Red) to HSV(Hue-Saturation-Value) color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Iterate over predictions
        for detection in detections:
            # Get the HSV image of the founded object
            img_box = img_hsv[int(detection["box_points"][1]):int(detection["box_points"][3]),
                              int(detection["box_points"][0]):int(detection["box_points"][2])]

            # Set range for red color and
            # define mask
            red_lower_lower = np.array([0, 49, 59], np.uint8)
            red_lower_upper = np.array([10, 49, 59], np.uint8)
            red_mask_lower = cv2.inRange(img_box, red_lower_lower, red_lower_upper)

            red_lower = np.array([169, 49, 59], np.uint8)  # possible values: 130, 41, 82 - 161, 155, 84 - 169, 49, 59
            red_upper = np.array([180, 255, 255], np.uint8)
            red_mask_upper = cv2.inRange(img_box, red_lower, red_upper)

            red_mask = red_mask_lower + red_mask_upper

            # Set range for green color and
            # define mask
            green_lower = np.array([25, 52, 72], np.uint8)
            green_upper = np.array([90, 255, 255], np.uint8)
            green_mask = cv2.inRange(img_box, green_lower, green_upper)

            # Set range for orange color and
            # define mask
            orange_lower = np.array([10, 50, 50], np.uint8)
            orange_upper = np.array([15, 255, 255], np.uint8)
            orange_mask = cv2.inRange(img_box, orange_lower, orange_upper)

            # Morphological Transform, Dilation
            # for each color and bitwise_and operator
            # between imageFrame and mask determines
            # to detect only that particular color
            kernel = np.ones((5, 5), "uint8")

            # For red color
            red_mask = cv2.dilate(red_mask, kernel)
            res_red = cv2.bitwise_and(img_box, img_box, mask=red_mask)

            # For green color
            green_mask = cv2.dilate(green_mask, kernel)
            res_green = cv2.bitwise_and(img_box, img_box, mask=green_mask)

            # For orange color
            orange_mask = cv2.dilate(orange_mask, kernel)
            res_orange = cv2.bitwise_and(img_box, img_box, mask=orange_mask)

            # Create contour to track red color
            # In some versions of OpenCV library 'findContours' returns three value; a modified image,
            # the contours and hierarchy. This throws 'ValueError: too many values to unpack (expected 2)'
            # exception. Add a variable 'x' before 'contours': x, contours, hierarchy = cv2.findContours(..)
            contours, hierarchy = cv2.findContours(red_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            max_red_area = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                max_red_area += area

            # Create contour to track green color
            contours, hierarchy = cv2.findContours(green_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            max_green_area = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                max_green_area += area

            # Create contour to track orange color
            contours, hierarchy = cv2.findContours(orange_mask,
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            max_orange_area = 0
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                max_orange_area += area

            # Draw Rectangle with the detected coordinates
            cv2.rectangle(img, (detection["box_points"][0], detection["box_points"][1]),
                          (detection["box_points"][2], detection["box_points"][3]), color=(0, 135, 0), thickness=2)

            # Founded object is mostly red
            if max_red_area > max_orange_area and max_red_area > max_green_area:
                if detection["name"] == 'sise-kola':
                    cv2.putText(img, 'coca-cola şişe ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), thickness=2)  # Write the prediction class on image
                    coca_cola_sise_counter += 1
                elif detection["name"] == 'kutu-kola':
                    cv2.putText(img, 'coca-cola kutu ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), thickness=2)  # Write the prediction class on image
                    coca_cola_kutu_counter += 1
                elif detection["name"] == 'cam-kola':
                    cv2.putText(img, 'coca-cola cam ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), thickness=2)  # Write the prediction class on image
                    coca_cola_cam_counter += 1
            
			# Founded object is mostly orange
            elif max_orange_area > max_red_area and max_orange_area > max_green_area:
                if detection["name"] == 'sise-kola':
                    cv2.putText(img, 'fanta şişe ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), thickness=1)  # Write the prediction class on image
                    fanta_sise_counter += 1
                elif detection["name"] == 'kutu-kola':
                    cv2.putText(img, 'fanta kutu ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), thickness=1)  # Write the prediction class on image
                    fanta_kutu_counter += 1
                elif detection["name"] == 'cam-kola':
                    cv2.putText(img, 'fanta cam ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), thickness=1)  # Write the prediction class on image
                    fanta_cam_counter += 1

            # Founded object is mostly green
            elif max_green_area > max_orange_area and max_green_area > max_red_area:
                if detection["name"] == 'sise-kola':
                    cv2.putText(img, 'sprite şişe ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), thickness=1)  # Write the prediction class on image
                    sprite_sise_counter += 1
                elif detection["name"] == 'kutu-kola':
                    cv2.putText(img, 'sprite kutu ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), thickness=1)  # Write the prediction class on image
                    sprite_kutu_counter += 1
                elif detection["name"] == 'cam-kola':
                    cv2.putText(img, 'sprite cam ' + '%.3f' % detection["percentage_probability"],
                                (detection["box_points"][0], detection["box_points"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), thickness=1)  # Write the prediction class on image
                    sprite_cam_counter += 1
            
        cv2.putText(img, 'sprite: {0} fanta: {1} coca-cola kutu: {2}'.format(str(sprite_kutu_counter),
                                                                             str(fanta_kutu_counter),
                                                                             str(coca_cola_kutu_counter)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 120), thickness=1)

        cv2.imwrite('C:/results/' + filename, img)


def extract_timestamp(filename):
    timestamp = filename.split('_')[1]
    return timestamp


def extract_mac(filename):
    mac = filename.split('_')[0]
    mac = mac.split('-')[0]
    return mac


def extract_shelfno(filename):
    shelfno = filename.split('_')[0]
    shelfno = shelfno.split('-')[1]
    return shelfno


# Log configuration
logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %('
                                                                                  'message)s', datefmt='%d-%b-%y '
                                                                                                       '%H:%M:%S')

# ----- Load Model -----
start = time.time()
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)  # pre-trained custom YOLOv3 model
detector.setJsonPath(json_path)  # custom detection configuration file
detector.loadModel()  # loading model from saved files
print('Model loaded.')
print("--- %s seconds ---" % (time.time() - start))
# ----------------------

while True:
    start = time.time()

    file_names = os.listdir(input_image_path)  # get file names from input image folder
    if len(file_names) <= 1:
        logging.error('The folder is empty.')
        print('The folder is empty.')

    else:
        file_names.remove('detected.jpg')  # remove detected.jpg from the file names list

        # Sort and group file names according to their mac
        file_names.sort(key=extract_mac)
        file_names_grouped_by_mac = [list(it) for k, it in groupby(file_names, extract_mac)]

        for i in range(len(file_names_grouped_by_mac)):
            # Shelf info for shelf 1, 2, and 3 for each cooler
            is_shelf_available = [False, False, False]

            fanta_kutu_counter = 0
            fanta_sise_counter = 0
            fanta_cam_counter = 0
            coca_cola_kutu_counter = 0
            coca_cola_sise_counter = 0
            coca_cola_cam_counter = 0
            sprite_kutu_counter = 0
            sprite_sise_counter = 0
            sprite_cam_counter = 0

            # Get the mac address of the cooler
            mac = extract_mac(file_names_grouped_by_mac[i][0])

            # Sort and group file names according to their shelf number for each group
            file_names_grouped_by_mac[i].sort(key=extract_shelfno)
            file_names_grouped_by_shelf = [list(it) for k, it in groupby(file_names_grouped_by_mac[i], extract_shelfno)]

            for j in range(len(file_names_grouped_by_shelf)):
                # Get the shelf number
                shelf_number = extract_shelfno(file_names_grouped_by_shelf[j][0])

                # Sort file names according to their time stamp --in case of multiple images for one shelf
                file_names_grouped_by_shelf[j].sort(key=extract_timestamp,
                                                    reverse=True)  # get the latest image of the shelf

                file_name = file_names_grouped_by_shelf[j][0]

                # Get stock info about the shelf
                get_stock_info(file_name)

                is_shelf_available[int(shelf_number) - 1] = True

            if False in is_shelf_available:  # one of the shelves is not available
                indexes = [i for i, x in enumerate(is_shelf_available) if not x]
                print('The shelves ' + str([x+1 for x in indexes]) + ' are not available for cooler ' + str(mac) + '.')
                logging.error('The shelves ' + str([x+1 for x in indexes]) + ' are not available for cooler ' + str(mac) + '.')
            # TODO: handle shelf not available situation
            else:
                print('----------------------')
                print('sprite: kutu: {0} şişe: {1} cam: {2}'.format(str(sprite_kutu_counter),
                                                                    str(sprite_sise_counter),
                                                                    str(sprite_cam_counter)))
                print('fanta: kutu: {0} şişe: {1} cam: {2}'.format(str(fanta_kutu_counter),
                                                                   str(fanta_sise_counter),
                                                                   str(fanta_cam_counter)))
                print('coca-cola: kutu: {0} şişe: {1} cam: {2}'.format(str(coca_cola_kutu_counter),
                                                                       str(coca_cola_sise_counter),
                                                                       str(coca_cola_cam_counter)))

                # JSON
                json_data = "[{'ProductID': '1', 'Quantity': %d}, {'ProductID': '3', 'Quantity': %d}, {'ProductID': '4', 'Quantity': %d}]" \
                            % (coca_cola_kutu_counter, fanta_kutu_counter, sprite_kutu_counter)

                try:
                    # Making a POST request
                    r = requests.post(url,
                                      data={'mac': mac, 'temp': temp, 'latitude': latitude, 'longitude': longitude,
                                            'json': json_data})
                except:
                    logging.error('Exception occurred.', exc_info=True)
                    print('Error! Please check your network connection.')
                else:
                    print('Connection successful.')

    duration = time.time() - start
    print("--- %s seconds ---" % duration)

    sleep_count = 40 - duration
    if sleep_count >= 0:
        sleep(sleep_count)
    else:
        sleep(40 - (-sleep_count % 40))
