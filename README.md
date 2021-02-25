# onnxruntime-nim
[onnxruntime C Api](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h) wrapped for nim

## Install

### 1. Install Onnxruntime
 * Go to [Onnxruntime Releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.6.0)
 * Choose source code/compiled binaries for your system, such as the one for [Linux Cpu 1.6.0](https://github.com/microsoft/onnxruntime/releases/download/v1.6.0/onnxruntime-linux-x64-1.6.0.tgz)
 * Unzip the file, copy ./include and ./lib to the right places
 
   For example, on Linux, you may copy ./lib to /usr/lib/x86_64-linux-gn/, and ./include/ to /usr/include/
   Or just put ./include ./lib into your project; as long as you and your code know where they are.

### 2. Install Onnxruntime C Api for nim
  ```nim
  git clone https://github.com/YesDrX/onnxruntime-nim
  cd onnxruntime-nim
  nimble install
  ```

### 3. Sample Code
* [C_Api_Sample.nim](https://github.com/YesDrX/onnxruntime-nim/blob/main/sample/C_Api_Sample.nim) is a direct translation from [C_Api_Sample.cpp](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)
