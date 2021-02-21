FROM reizouko/installed-tensorflow-gpu1.15.0rc3

WORKDIR /usr/src/app

RUN pip install opencv-python \
        jsonpickle \
        flask \
        pyyaml \
        networkx 

RUN apt install -y --no-install-recommends libgl1-mesa-glx

CMD python /usr/src/app/init_http.py

COPY . .