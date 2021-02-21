import os
import jsonpickle
import time
import numpy as np
import cv2 as cv
import logging
import yaml
from cloud_server import CServer
server = None
def single_image():
    start_time = time.time()

    args = None
    global server
    server = CServer(args, nframes=None)
    file_data = '0000000000.png'
    result = server.perform_single_image(file_data)
    # build a response dict to send back to client
    # response = {'message': 'image received. size={}x{}'.format(image.shape[1], image.shape[0])}
    # encode response using jsonpickle
    # response_pickled = jsonpickle.encode(result)
    end_time = time.time()
    print(end_time-start_time)
    # return Response(response=response_pickled, status=200, mimetype="application/json")
    return 'OK!'

single_image()