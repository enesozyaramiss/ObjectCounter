# ObjectCounter
This code is a Python script that performs object detection and sends the quantities of detected objects to a server via a POST request. Here's a breakdown of the general workflow:

![image](https://github.com/enesozyaramiss/ObjectCounter/assets/62839938/ae85f85e-0ada-4541-9b42-855978d8b21a)


Library Imports: The code imports several libraries to support various functionalities, including image processing (OpenCV), object detection (scipy), and making HTTP requests (requests).

Functions: The inference function performs object detection on an image, detecting objects and their locations. The extract_shelfno, extract_timestamp, and extract_mac functions are used to extract information from file names.

Main Loop: Within an infinite loop, image files in a specific folder are processed. In each iteration, images from Raspberry Pis with a specific MAC address are grouped, and object detection is performed for each. The detected object counts are summed up, and this data is converted into a JSON format and sent to the server via a POST request.

HTTP POST Request: The requests.post function is used to make an HTTP POST request to the server. This request sends JSON data containing the MAC address of the Raspberry Pis, temperature, latitude, longitude, and quantities of detected objects.

Error Handling: The code uses try-except blocks to handle possible errors and prints appropriate log messages in case of errors.

File Deletion: There is a loop for deleting files after they are processed, but currently, it's commented out, meaning the code doesn't delete these files. You can uncomment these lines to enable file deletion.

![image](https://github.com/enesozyaramiss/ObjectCounter/assets/62839938/0b315cd1-d759-4f9b-8df7-26e849311765)


JSON Data: The quantities of all detected objects are converted into a JSON format, and this JSON data is sent to the server.

Communication Check: After communicating with the server, the HTTP status code is checked, and if the operation is successful, a "Connection Successful" message is printed.

This code constitutes a system capable of performing object detection and sending the quantities of detected objects to a server. However, you can customize the code according to specific requirements or enhancements you might have.
