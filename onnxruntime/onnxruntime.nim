import ./onnxruntimeApi/onnxruntime_c_api

let 
    apiBase = OrtGetApiBase()
    onxApi* = apiBase.GetApi(ORT_API_VERSION)

echo "Using Onnxruntime C Api : " & $apiBase.GetVersionString()

export onnxruntime_c_api