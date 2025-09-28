@echo off

REM The correct path to the Python executable in the venv is .\venv\Scripts\python.exe
REM The rest of the script calls the onnx2trt.py file in the scripts folder

REM warping+spade model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\warping_spade-fix.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\warping_spade-fix.onnx

REM landmark model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\landmark.onnx

REM motion_extractor model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\motion_extractor.onnx -p fp32
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\motion_extractor.onnx -p fp32

REM face_analysis model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\retinaface_det_static.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\face_2dpose_106_static.onnx

REM appearance_extractor model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\appearance_feature_extractor.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\appearance_feature_extractor.onnx

REM stitching model
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_eye.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_onnx\stitching_lip.onnx

.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching_eye.onnx
.\venv\Scripts\python.exe scripts\onnx2trt.py -o .\checkpoints\liveportrait_animal_onnx\stitching_lip.onnx
