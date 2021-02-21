import os
from flask import Flask, request, Response
import jsonpickle
import time
import sys
import numpy as np
import cv2 as cv
import logging
import signal
from cloud_server import CServer
import subprocess
port = sys.argv[1]
logging.getLogger("init")
# Initialize the Flask application
app = Flask(__name__)
cserver = None

def kill_process(pid):
    time.sleep(1)
    os.kill(pid, signal.SIGKILL)

@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    # args = yaml.load(request.data, Loader=yaml.SafeLoader)
    args = None
    global cserver
    logging.basicConfig(format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s", level="INFO")
    if not cserver:
        cserver = CServer(args, nframes=None)
        # os.makedirs("cserver_temp", exist_ok=True)
        # os.makedirs("cserver_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        # server.reset_state(int(args["nframes"]))
        return "Reset"

@app.route("/gpu_monitor", methods=["POST"])
def gpu_monitor():
    index = request.form['index']
    func = request.form['function']
    cmd = 'nvidia-smi dmon -i 0 -d 5 -s u -c 100 -f results/gpu_{}_{}.log'.format(func, index)    # 这里需要修改
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    return "gpu_monitor started"

@app.route('/image', methods=['POST'])
def single_image():
    file_data = request.files["media"]
    start_time = time.time()
    # single image
    point = request.form['point']
    results = cserver.perform_single_image(file_data, point, binary_image=True)
    # build a response dict to send back to client
    response = {'message': 'ok'}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    end_time = time.time()
    print('process image:', end_time - start_time)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    # return 'OK!'


@app.route('/video', methods=['POST'])
def analyze_video():
    file_data = request.files["media"]
    # the data is video transmitted from client
    start_time = time.time()
    results = cserver.perform_video_np(file_data)
    # build a response dict to send back to client
    # response = {'message': 'image received. size={}x{}'.format(image.shape[1], image.shape[0])}
    # encode response using jsonpickle
    response = {'message': str(results['results'])}
    response_pickled = jsonpickle.encode(results)
    end_time = time.time()
    # print(end_time - start_time)
    print(response)
    # return Response(response=response_pickled, status=200, mimetype="application/json")
    return 'OK!'


@app.route('/create', methods=['POST'])
def create_app():
    file_path = '0000000000.png'
    result = cserver.perform_single_image(file_path, binary_image=False)
    response = {'message': 'Creating the model instance is completed'}
    print(response)
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/kill', methods=['POST'])
def kill_app():
    # response = {'message': 'killing the instance'}
    # response_pickled = jsonpickle.encode(response)
    # return Response(response=response_pickled, status=200, mimetype="application/json")
    # print('waiting to kill the instance')



    response = {'message': 'killing the model instance'}
    # print(response)
    logging.info("killing the instance")
    response_pickled = jsonpickle.encode(response)
    from multiprocessing import Process
    pid = os.getpid()
    Process(target=kill_process, args=(pid, )).start()
    return Response(response=response_pickled, status=200, mimetype="application/json")




# start flask app
app.run(host="0.0.0.0", port=int(port))