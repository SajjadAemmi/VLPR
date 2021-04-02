# LPRS: License Plate Recognition System 

Deep Learning based License Plate Detection and OCR

## Inference

### MacOS M1
Run following commands:
```
mkdir build
cmake -S . -B build -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build build
./build/LPRS
```