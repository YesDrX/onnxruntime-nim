import onnxruntime
import sequtils
import strformat

let
    api = OrtGetApiBase().GetApi(ORT_API_VERSION)
    model_path = "squeezenet.onnx".cstring

proc `$`(strPtr: ptr cstring): string=
    return $(cast[cstringArray](strPtr).cstringArrayToSeq)

proc checkStatus(status: OrtStatusPtr) =
    if status != nil:
        let msg = api.GetErrorMessage(status)
        echo fmt"[ERROR] : {msg}"
        api.ReleaseStatus(status)
        quit 1

when isMainModule:
    var
        env : ptr OrtEnv
        session_options : ptr OrtSessionOptions
        session : ptr OrtSession
        allocator : ptr OrtAllocator
        num_input_nodes : csize_t
        input_node_names : seq[cstring]
        input_name : cstring
        typeinfo : ptr OrtTypeInfo
        tensor_info : ptr OrtTensorTypeAndShapeInfo
        dataType : ONNXTensorElementDataType
        num_dims : csize_t
        input_node_dims : seq[int]
        input_tensor_size : int = 1

    api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test".cstring, env.addr).checkStatus
    api.CreateSessionOptions(session_options.addr).checkStatus
    api.SetIntraOpNumThreads(session_options, 1).checkStatus
    api.SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC).checkStatus

    echo fmt"Using Onnxruntime C API"

    api.CreateSession(env, model_path, session_options, session.addr).checkStatus
    api.GetAllocatorWithDefaultOptions(allocator.addr).checkStatus
    api.SessionGetInputCount(session, num_input_nodes.addr).checkStatus
    
    input_node_names = newSeq[cstring](num_input_nodes)
    echo fmt"Number of inputs = {num_input_nodes}"

    for i in 0 .. num_input_nodes.int - 1:
        api.SessionGetInputName(session, 0.csize_t, allocator, input_name.addr).checkStatus
        echo fmt"Input {i} : name= {input_name}"
        input_node_names[i] = input_name

        api.SessionGetInputTypeInfo(session, i.csize_t, typeinfo.addr).checkStatus
        api.CastTypeInfoToTensorInfo(typeinfo, tensor_info.addr).checkStatus
        api.GetTensorElementType(tensor_info, dataType.addr).checkStatus
        
        echo fmt"Input {i} : type = {dataType}"
        
        api.GetDimensionsCount(tensor_info, num_dims.addr).checkStatus
        echo fmt"Input {i} : num_dims = {num_dims.int}"
        
        input_node_dims = newSeq[int](num_dims)
        api.GetDimensions(tensor_info, cast[ptr int64](input_node_dims[0].addr), num_dims).checkStatus
        for j in 0 .. num_dims.int - 1:
            echo fmt"Input {i} : dim {j} = {input_node_dims[j]}"        

        api.ReleaseTypeInfo(typeinfo)

    for i in 0 .. num_dims-1:
        input_tensor_size *= input_node_dims[i]
    
    var
        input_tensor_values = newSeq[cfloat](input_tensor_size)
        output_node_names = @["softmaxout_1".cstring]
        memory_info : ptr OrtMemoryInfo
        input_tensor : ptr OrtValue
        output_tensor  : ptr OrtValue
        is_tensor : cint
        floatarrPtr: ptr cfloat

    output_node_names.add("softmaxout_1".cstring)
    # initialize input data with values in [0.0, 1.0]
    for i in 0 .. input_tensor_size - 1:
        input_tensor_values[i] = i.float / (input_tensor_size + 1).float
    
    # create input tensor object from data dim_values
    api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, memory_info.addr).checkStatus
    api.CreateTensorWithDataAsOrtValue(memory_info, cast[pointer](input_tensor_values[0].addr), (input_tensor_size * cfloat.sizeof).csize_t, cast[ptr int64](input_node_dims[0].addr), 4.csize_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensor.addr).checkStatus
    api.IsTensor(input_tensor, is_tensor.addr).checkStatus
    assert is_tensor == 1.cint
    api.ReleaseMemoryInfo(memory_info)

    # score model & input tensor, get back output tensor
    api.Run(session, cast[ptr OrtRunOptions](0), cast[ptr cstring](input_node_names[0].addr), cast[ptr ptr OrtValue](input_tensor.addr), 1.csize_t, cast[ptr cstring](output_node_names[0].addr), 1.csize_t, output_tensor.addr).checkStatus
    api.IsTensor(output_tensor, is_tensor.addr).checkStatus
    assert is_tensor == 1.cint

    api.GetTensorMutableData(output_tensor, cast[ptr pointer](floatarrPtr.addr)).checkStatus

    var floatarr = cast[ptr UncheckedArray[cfloat]](floatarrPtr)
    for i in 0 .. 4:
        echo fmt"[Class {i}] : {floatarr[i] : 0.6f}" 

    api.ReleaseValue(output_tensor)
    api.ReleaseValue(input_tensor)
    api.ReleaseSession(session)
    api.ReleaseSessionOptions(session_options)
    api.ReleaseEnv(env)   

    echo "Done!"