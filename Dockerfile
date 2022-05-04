FROM sajjadaemmi/opencv-cuda

WORKDIR /workspace/LPR
COPY . .

WORKDIR /workspace/LPR/python
RUN pip3 install -r requirements.txt

WORKDIR /workspace/LPR/python/source/lanms
RUN make

WORKDIR /workspace/LPR