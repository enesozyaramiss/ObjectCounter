import logging
import os
import os.path
import time
from time import sleep
from itertools import groupby
from scipy import misc
import cv2 as cv
import numpy as np
import requests  # to use HTTP POST request

input_image_path = ""  # folder path
raspberry_mac_addresses = [""serverprocess_6","serverprocess_7","serverprocess_8"]
raspberry_mac_status = [True, True]

# HTTP POST request variables
url = ''
#url = ''
mac = ''
temp = 0
latitude = 0
longitude = 0
json_data = ""
# -------------------------------
eneslist = []
# Log configuration
logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %('
                                                                                  'message)s', datefmt='%d-%b-%y '
                                                                                                       '%H:%M:%S')
def inference(img, shelf_number):	
    kolacounter = 0 
    fantacounter = 0
    spritecounter = 0
    net = cv.dnn_DetectionModel('custom-yolov4-detector.cfg',
                                'custom-yolov4-detector_best.weights')
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
	
    with open('obj.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
		
    classes, confidences, boxes = net.detect(img, confThreshold=0.1, nmsThreshold=0.4)
    if len(confidences) > 0.4:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = '%.2f' % confidence
            label = '%s: %s' % (names[classId], label)
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            #print(shelf_number)
            count_status = True
            if shelf_number == '1':
                if 0 < 700 - top < 10 :
                    eneslist.append((top,left))
                    print(eneslist)
                if classId == 1 and (top < 700 and left < 2592) and (width < 300 and height < 200):
                   # print(width, height)
                    kolacounter +=1
                    cv.rectangle(img, box, color=(255, 0, 0 ), thickness=3)
                elif classId == 0 and (top < 700 and left < 2592) and (width < 300 and height < 200):
                    #print(width, height)
                    fantacounter +=1
                    cv.rectangle(img, box, color=(0, 0 , 255), thickness=3)
                        
                elif classId == 2 and (top < 700 and left < 2592) and (width < 300 and height < 200):
                    #print(width, height)
                    spritecounter +=1
                    cv.rectangle(img, box, color=(0,255 , 0), thickness=3)
            if shelf_number == '2':
                for item in eneslist:
                    listtop, listleft = item
                    if 0 < top - listtop < 10 or 0 < listtop - top < 10:
                        count_status = False
                        break
                if count_status:
                    if classId == 1 and (top < 1133 and left < 2592) and (width < 300 and height < 200):
                       # print(width, height)
                        kolacounter +=1
                        cv.rectangle(img, box, color=(255, 0, 0 ), thickness=3)
                    elif classId == 0 and (top < 1133 and left < 2592) and (width < 300 and height < 200):
                        #print(width, height)
                        fantacounter +=1
                        cv.rectangle(img, box, color=(0, 0 , 255), thickness=3)
                            
                    elif classId == 2 and (top < 1133 and left < 2587) and (width < 300 and height < 200):
                        #print(width, height)
                        spritecounter +=1
                        cv.rectangle(img, box, color=(0,255 , 0), thickness=3)
            


    return kolacounter, fantacounter, spritecounter

def extract_shelfno(filename):
	mac = filename.split('_')[0]
	shelfno = mac.split('-')[1]
	return shelfno
   
def extract_timestamp(filename):
    timestamp = filename.split('_')[1]
    return timestamp

def extract_mac(filename):
    mac = filename.split('_')[0]
    mac = mac.split('-')[0]
    return mac
   
while True:
    totalkolacounter = 0
    totalfantacounter = 0 
    totalspritecounter = 0
    print("---------------------------")

    file_names = os.listdir(input_image_path)  # get file names from input image folder
    if len(file_names) <= 0:
        logging.error('The folder is empty.')
        print('The folder is empty.')
        sleep(1)

    else:
		# Sort and group file names according to their mac
        file_names.sort(key=extract_mac)
        file_names_grouped_by_mac = [list(it) for k, it in groupby(file_names, extract_mac)]
        print("range1:"+str(len(file_names_grouped_by_mac)))
        for i in range(len(file_names_grouped_by_mac)):
			# Get the mac address of the cooler
            mac = extract_mac(file_names_grouped_by_mac[i][0])

			# Sort and group file names according to their shelf number for each group
            file_names_grouped_by_mac[i].sort(key=extract_shelfno)
            file_names_grouped_by_shelf = [list(it) for k, it in groupby(file_names_grouped_by_mac[i], extract_shelfno)]
            print("range2:"+str(len(file_names_grouped_by_shelf)))
            for j in range(len(file_names_grouped_by_shelf)):
				# Get the shelf number
                shelf_number = extract_shelfno(file_names_grouped_by_shelf[j][0])

				# Sort file names according to their time stamp --in case of multiple images for one shelf
                file_names_grouped_by_shelf[j].sort(key=extract_timestamp,
                                                    reverse=True)  # get the latest image of the shelf

                file_name = file_names_grouped_by_shelf[j][0]
				
                print(input_image_path + file_name)
                path = input_image_path + file_name
                
                img = cv.imread(path)
                kolacounter, fantacounter, spritecounter = inference(img, shelf_number)
                totalkolacounter += kolacounter
                totalfantacounter += fantacounter
                totalspritecounter += spritecounter
                print(totalkolacounter)
                print(totalfantacounter)
                print(totalspritecounter)
                cv.imwrite("enes.jpg",img)
		# JSON
        json_data = "[{'ProductID': '1', 'Quantity': '%d'}, {'ProductID': '3', 'Quantity': '%d'}, {'ProductID': '4', 'Quantity': '%d'}]" \
                            % (totalkolacounter, totalfantacounter, totalspritecounter)

        try:
			# Making a POST request
            r = requests.post(url,
                              data={'mac': mac, 'temp': temp, 'latitude': latitude, 'longitude': longitude,
                                    'json': json_data})
            #Dosyada Okunan Fotoğrafları Silme
			# for i in range(len(file_names)):
				# os.remove(input_image_path + file_names[i])
				# print(str(file_names[i]) + " dosyası silinmiştir.")
        except Exception as e:
            logging.error('Exception occurred.', exc_info=True)
            print('Hata! Lutfen Internetinizi Kontrol Edin.')
			#print(str(e))						
        except Exception as e:
            logging.error('Exception occurred.', exc_info=True)
            print('Hata! Lutfen Internetinizi Kontrol Edin.')
        else:
            print(r.status_code)
            print('Baglanti Basarili.')