import requests
import time
import cv2
import yaml
import subprocess
addr = 'http://166.111.9.92:5000'
session = requests.Session()

def init():
# prepare headers for http request
    start_time = time.time()
    response = session.post(addr+"/init")
    print(response.text)
    end_time = time.time()
    # print(end_time-start_time)

def send_image():
    encoded_image_path = '/home/rch/MPS_test/dataset/trafficcam_1/src/0000000000.png'
    image_to_send = {"media": open(encoded_image_path, "rb")}
    # send http request with image and receive response
    start_time = time.time()
    url = addr + '/image'
    response = session.post(url, files=image_to_send)
    # decode response
    # print(json.loads(response.text))
    print(response.text)
    end_time = time.time()
    # print(end_time-start_time)

def create_model():
    url = addr + '/create'
    response = session.post(url)
    print(response.text)

def kill_app():
    url = addr + '/kill'
    response = session.post(url)
    print(response.text)

# send a video
def send_video():

    encoded_video_path = 'video/output000.mp4'
    video_to_send = {"media": open(encoded_video_path, "rb")}
    # send http request with image and receive response
    start_time = time.time()
    url = addr + '/video'
    response = session.post(url, files=video_to_send)
    # decode response
    # print(json.loads(response.text))
    print(response.text)
    end_time = time.time()
    print(end_time-start_time)

init()
create_model()
send_image()
kill_app()

# send_video()


# expected output: {u'message': u'image received. size=124x124'}


