# onnxruntime-nim
[onnxruntime C Api](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h) wrapped for nim

* Wrapped C Api is generated using [c2nim](https://github.com/nim-lang/c2nim)
* [Onnxruntime Home Page](https://www.onnxruntime.ai/)
* [Onnxruntime Github](https://github.com/microsoft/onnxruntime)
* Header file [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h)

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
* Assume you have both Onnxruntime library and Onnxruntime-nim installed
```nim
cd ./sample
nim c --run C_Api_Sample.nim
```
* Output
```
Using Onnxruntime C Api : 1.6.0
WARNING: Since openmp is enabled in this build, this API cannot be used to configure intra op num threads. Please use the openmp environment variables to control the number of threads.
Using Onnxruntime C API
Number of inputs = 1
Input 0 : name= data_0
Input 0 : type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
Input 0 : num_dims = 4
Input 0 : dim 0 = 1
Input 0 : dim 1 = 3
Input 0 : dim 2 = 224
Input 0 : dim 3 = 224
[Class 0] :  0.000045
[Class 1] :  0.003846
[Class 2] :  0.000125
[Class 3] :  0.001180
[Class 4] :  0.001317
Done!
```

```
