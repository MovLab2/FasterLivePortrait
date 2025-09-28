#  NVIDIA Blackwell RTX 5060 Ti Support

## Complete Working Configuration
- **GPU**: NVIDIA GeForce RTX 5060 Ti (Blackwell sm_120)
- **TensorRT**: 10.13.3.9
- **Status**:  Fully Working with TensorRT Acceleration

## All Fixes Applied
### 1. Environment Setup
- NumPy 1.26.4 (fixed ONNX Runtime compatibility)
- Updated TensorRT conversion scripts
- Fixed motion extractor output order

### 2. Key File Changes
- \scripts/onnx2trt.py\ - Updated TensorRT API compatibility
- \src/models/motion_extractor_model.py\ - Fixed output order
- \webui.py\ - Kokoro error patches
- \equirements.txt\ - Working dependency versions

### 3. TensorRT Conversion
- All ONNX models successfully converted to TensorRT
- Proper output dimension preservation
- Blackwell GPU acceleration enabled

## Performance Results
- **Source Processing**: 6.71 it/s
- **Video Processing**: 5.41 it/s (587 frames in 1:48)
- **All 8 Models**: TensorRT accelerated successfully

## Usage
\\\ash
python webui.py --mode trt --host 0.0.0.0
\\\

## Verified Files
-  Updated conversion scripts
-  Fixed model processing code
-  Working requirement files
-  Kokoro error patches

**Complete working setup tested on RTX 5060 Ti - September 2024**
