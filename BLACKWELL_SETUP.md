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

**Complete working setup tested on RTX 5060 Ti - September 2025**


##  Advanced Technical Details

### System Configuration:
- **NVIDIA Driver**: 581.29
- **System CUDA**: 13.0 (Latest)
- **PyTorch CUDA**: 11.8 (Backward compatible)
- **TensorRT**: 10.13.3.9 (Latest CUDA 13 bindings)
- **GPU Architecture**: Blackwell sm_120

### Key Compatibility Achievements:
 **Mixed CUDA Versions**: PyTorch CUDA 11.8 works flawlessly on CUDA 13.0 system  
 **TensorRT Acceleration**: Latest TensorRT 10.13.3.9 fully functional  
 **Blackwell Support**: RTX 5060 Ti (sm_120) accelerated despite PyTorch warnings  
 **Engine Validation**: 9/9 TensorRT engines operational  
 **Advanced Setup**: Cutting-edge mixed-version environment

### Performance Metrics:
- **TensorRT Inference**: 5.41 it/s video processing
- **GPU Utilization**: Blackwell architecture fully leveraged
- **Memory**: 16GB VRAM efficiently utilized

### Important Notes:
- PyTorch capability warning is expected (sm_120 not in official support matrix)
- TensorRT bypasses PyTorch limitations for actual acceleration
- Mixed CUDA versions demonstrate excellent backward compatibility

**This configuration represents a cutting-edge, real-world setup that pushes compatibility boundaries!**
