{.deadCodeElim: on.}

when defined(windows):
  const libname* = "onnxruntime.dll"
elif defined(macosx):
  const libname* = "libonnxruntime.dylib"
else:
  const libname* = "libonnxruntime.so"

const
  ORT_API_VERSION* = 6

type
  ONNXTensorElementDataType* {.size: sizeof(cint).} = enum
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
  
  ONNXType* {.size: sizeof(cint).} = enum
    ONNX_TYPE_UNKNOWN, ONNX_TYPE_TENSOR, ONNX_TYPE_SEQUENCE, ONNX_TYPE_MAP,
    ONNX_TYPE_OPAQUE, ONNX_TYPE_SPARSETENSOR
  
  OrtLoggingLevel* {.size: sizeof(cint).} = enum
    ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_INFO, ORT_LOGGING_LEVEL_WARNING,
    ORT_LOGGING_LEVEL_ERROR, ORT_LOGGING_LEVEL_FATAL
  
  OrtErrorCode* {.size: sizeof(cint).} = enum
    ORT_OK, ORT_FAIL, ORT_INVALID_ARGUMENT, ORT_NO_SUCHFILE, ORT_NO_MODEL,
    ORT_ENGINE_ERROR, ORT_RUNTIME_EXCEPTION, ORT_INVALID_PROTOBUF,
    ORT_MODEL_LOADED, ORT_NOT_IMPLEMENTED, ORT_INVALID_GRAPH, ORT_EP_FAIL

type
  OrtEnv* {.bycopy.} = object

  OrtStatus* {.bycopy.} = object

  OrtMemoryInfo* {.bycopy.} = object

  OrtIoBinding* {.bycopy.} = object

  OrtSession* {.bycopy.} = object

  OrtValue* {.bycopy.} = object

  OrtRunOptions* {.bycopy.} = object

  OrtTypeInfo* {.bycopy.} = object

  OrtTensorTypeAndShapeInfo* {.bycopy.} = object

  OrtSessionOptions* {.bycopy.} = object

  OrtCustomOpDomain* {.bycopy.} = object

  OrtMapTypeInfo* {.bycopy.} = object

  OrtSequenceTypeInfo* {.bycopy.} = object

  OrtModelMetadata* {.bycopy.} = object

  OrtThreadPoolParams* {.bycopy.} = object

  OrtThreadingOptions* {.bycopy.} = object

  OrtArenaCfg* {.bycopy.} = object

  OrtStatusPtr* = ptr OrtStatus

  OrtAllocator* {.bycopy.} = object
    version*: uint32
    Alloc*: proc (this: ptr OrtAllocator, size: csize_t): pointer {.cdecl.}
    Free*: proc (this: ptr OrtAllocator, p: pointer) {.cdecl.}
    Info*: proc (this: ptr OrtAllocator): ptr OrtMemoryInfo {.cdecl.}

  OrtLoggingFunction* = proc (param: pointer, severity: OrtLoggingLevel, category: cstring, logid: cstring,code_location: cstring, message: cstring) {.cdecl.}

  GraphOptimizationLevel* {.size: sizeof(cint).} = enum
    ORT_DISABLE_ALL = 0, ORT_ENABLE_BASIC = 1, ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99
  
  ExecutionMode* {.size: sizeof(cint).} = enum
    ORT_SEQUENTIAL = 0, ORT_PARALLEL = 1
  
  OrtLanguageProjection* {.size: sizeof(cint).} = enum
    ORT_PROJECTION_C = 0, ORT_PROJECTION_CPLUSPLUS = 1, ORT_PROJECTION_CSHARP = 2,
    ORT_PROJECTION_PYTHON = 3, ORT_PROJECTION_JAVA = 4, ORT_PROJECTION_WINML = 5,
    ORT_PROJECTION_NODEJS = 6

type
  OrtKernelInfo* {.bycopy.} = object

  OrtKernelContext* {.bycopy.} = object

  OrtAllocatorType* {.size: sizeof(cint).} = enum
    Invalid = -1, OrtDeviceAllocator = 0, OrtArenaAllocator = 1

  OrtMemType* {.size: sizeof(cint).} = enum
    OrtMemTypeCPUInput = -2, OrtMemTypeCPUOutput = -1, OrtMemTypeDefault = 0

  OrtCudnnConvAlgoSearch* {.size: sizeof(cint).} = enum
    EXHAUSTIVE, HEURISTIC, DEFAULT

  OrtCUDAProviderOptions* {.bycopy.} = object
    device_id*: cint
    cudnn_conv_algo_search*: OrtCudnnConvAlgoSearch
    cuda_mem_limit*: csize_t
    arena_extend_strategy*: cint
    do_copy_in_default_stream*: cint

  OrtOpenVINOProviderOptions* {.bycopy.} = object
    device_type*: cstring
    enable_vpu_fast_compile*: cuchar
    device_id*: cstring
    num_of_threads*: csize_t

const
  OrtMemTypeCPU = OrtMemTypeCPUOutput

type
  OrtApi* {.bycopy.} = object
    CreateStatus*: proc (code: OrtErrorCode, msg: cstring): ptr OrtStatus {.cdecl.}
    GetErrorCode*: proc (status: ptr OrtStatus): OrtErrorCode {.cdecl.}
    GetErrorMessage*: proc (status: ptr OrtStatus): cstring {.cdecl.}
    CreateEnv*: proc (logging_level: OrtLoggingLevel, logid: cstring,`out`: ptr ptr OrtEnv): OrtStatusPtr {.cdecl.}
    CreateEnvWithCustomLogger*: proc (logging_function: OrtLoggingFunction, logger_param: pointer, logging_level: OrtLoggingLevel, logid: cstring, `out`: ptr ptr OrtEnv): OrtStatusPtr {.cdecl.}
    EnableTelemetryEvents*: proc (env: ptr OrtEnv): OrtStatusPtr {.cdecl.}
    DisableTelemetryEvents*: proc (env: ptr OrtEnv): OrtStatusPtr {.cdecl.}
    CreateSession*: proc (env: ptr OrtEnv, model_path: cstring, options: ptr OrtSessionOptions, `out`: ptr ptr OrtSession): OrtStatusPtr {.cdecl.}
    CreateSessionFromArray*: proc (env: ptr OrtEnv, model_data: pointer, model_data_length: csize_t, options: ptr OrtSessionOptions, `out`: ptr ptr OrtSession): OrtStatusPtr {.cdecl.}
    Run*: proc (sess: ptr OrtSession, run_options: ptr OrtRunOptions, input_names: ptr cstring, input: ptr ptr OrtValue, input_len: csize_t, output_names1: ptr cstring, output_names_len: csize_t, output: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    CreateSessionOptions*: proc (options: ptr ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    SetOptimizedModelFilePath*: proc (options: ptr OrtSessionOptions, optimized_model_filepath: cstring): OrtStatusPtr {.cdecl.}
    CloneSessionOptions*: proc (in_options: ptr OrtSessionOptions, out_options: ptr ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    SetSessionExecutionMode*: proc (options: ptr OrtSessionOptions, execution_mode: ExecutionMode): OrtStatusPtr {.cdecl.}
    EnableProfiling*: proc (options: ptr OrtSessionOptions, profile_file_prefix: cstring): OrtStatusPtr {.cdecl.}
    DisableProfiling*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    EnableMemPattern*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    DisableMemPattern*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    EnableCpuMemArena*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    DisableCpuMemArena*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    SetSessionLogId*: proc (options: ptr OrtSessionOptions, logid: cstring): OrtStatusPtr {.cdecl.}
    SetSessionLogVerbosityLevel*: proc (options: ptr OrtSessionOptions, session_log_verbosity_level: cint): OrtStatusPtr {.cdecl.}
    SetSessionLogSeverityLevel*: proc (options: ptr OrtSessionOptions, session_log_severity_level: cint): OrtStatusPtr {.cdecl.}
    SetSessionGraphOptimizationLevel*: proc (options: ptr OrtSessionOptions,graph_optimization_level: GraphOptimizationLevel): OrtStatusPtr {.cdecl.}
    SetIntraOpNumThreads*: proc (options: ptr OrtSessionOptions, intra_op_num_threads: cint): OrtStatusPtr {.cdecl.}
    SetInterOpNumThreads*: proc (options: ptr OrtSessionOptions, inter_op_num_threads: cint): OrtStatusPtr {.cdecl.}
    CreateCustomOpDomain*: proc (domain: cstring, `out`: ptr ptr OrtCustomOpDomain): OrtStatusPtr {.cdecl.}
    CustomOpDomain_Add*: proc (custom_op_domain: ptr OrtCustomOpDomain, op: ptr OrtCustomOp): OrtStatusPtr {.cdecl.}
    AddCustomOpDomain*: proc (options: ptr OrtSessionOptions, custom_op_domain: ptr OrtCustomOpDomain): OrtStatusPtr {.cdecl.}
    RegisterCustomOpsLibrary*: proc (options: ptr OrtSessionOptions, library_path: cstring, library_handle: ptr pointer): OrtStatusPtr {.cdecl.}
    SessionGetInputCount*: proc (sess: ptr OrtSession, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    SessionGetOutputCount*: proc (sess: ptr OrtSession, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    SessionGetOverridableInitializerCount*: proc (sess: ptr OrtSession,`out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    SessionGetInputTypeInfo*: proc (sess: ptr OrtSession, index: csize_t, type_info: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    SessionGetOutputTypeInfo*: proc (sess: ptr OrtSession, index: csize_t, type_info: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    SessionGetOverridableInitializerTypeInfo*: proc (sess: ptr OrtSession,index: csize_t, type_info: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    SessionGetInputName*: proc (sess: ptr OrtSession, index: csize_t, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    SessionGetOutputName*: proc (sess: ptr OrtSession, index: csize_t, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    SessionGetOverridableInitializerName*: proc (sess: ptr OrtSession,index: csize_t, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    CreateRunOptions*: proc (`out`: ptr ptr OrtRunOptions): OrtStatusPtr {.cdecl.}
    RunOptionsSetRunLogVerbosityLevel*: proc (options: ptr OrtRunOptions, value: cint): OrtStatusPtr {.cdecl.}
    RunOptionsSetRunLogSeverityLevel*: proc (options: ptr OrtRunOptions, value: cint): OrtStatusPtr {.cdecl.}
    RunOptionsSetRunTag*: proc (a1: ptr OrtRunOptions, run_tag: cstring): OrtStatusPtr {.cdecl.}
    RunOptionsGetRunLogVerbosityLevel*: proc (options: ptr OrtRunOptions,`out`: ptr cint): OrtStatusPtr {.cdecl.}
    RunOptionsGetRunLogSeverityLevel*: proc (options: ptr OrtRunOptions,`out`: ptr cint): OrtStatusPtr {.cdecl.}
    RunOptionsGetRunTag*: proc (a1: ptr OrtRunOptions, `out`: ptr cstring): OrtStatusPtr {.cdecl.}
    RunOptionsSetTerminate*: proc (options: ptr OrtRunOptions): OrtStatusPtr {.cdecl.}
    RunOptionsUnsetTerminate*: proc (options: ptr OrtRunOptions): OrtStatusPtr {.cdecl.}
    CreateTensorAsOrtValue*: proc (allocator: ptr OrtAllocator, shape: ptr int64, shape_len: csize_t, `type`: ONNXTensorElementDataType, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    CreateTensorWithDataAsOrtValue*: proc (info: ptr OrtMemoryInfo, p_data: pointer,p_data_len: csize_t, shape: ptr int64, shape_len: csize_t,`type`: ONNXTensorElementDataType, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    IsTensor*: proc (value: ptr OrtValue, `out`: ptr cint): OrtStatusPtr {.cdecl.}
    GetTensorMutableData*: proc (value: ptr OrtValue, `out`: ptr pointer): OrtStatusPtr {.cdecl.}
    FillStringTensor*: proc (value: ptr OrtValue, s: ptr cstring, s_len: csize_t): OrtStatusPtr {.cdecl.}
    GetStringTensorDataLength*: proc (value: ptr OrtValue, len: ptr csize_t): OrtStatusPtr {.cdecl.}
    GetStringTensorContent*: proc (value: ptr OrtValue, s: pointer, s_len: csize_t, offsets: ptr csize_t, offsets_len: csize_t): OrtStatusPtr {.cdecl.}
    CastTypeInfoToTensorInfo*: proc (a1: ptr OrtTypeInfo, `out`: ptr ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.cdecl.}
    GetOnnxTypeFromTypeInfo*: proc (a1: ptr OrtTypeInfo, `out`: ptr ONNXType): OrtStatusPtr {.cdecl.}
    CreateTensorTypeAndShapeInfo*: proc (`out`: ptr ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.cdecl.}
    SetTensorElementType*: proc (a1: ptr OrtTensorTypeAndShapeInfo, `type`: ONNXTensorElementDataType): OrtStatusPtr {.cdecl.}
    SetDimensions*: proc (info: ptr OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_count: csize_t): OrtStatusPtr {.cdecl.}
    GetTensorElementType*: proc (a1: ptr OrtTensorTypeAndShapeInfo, `out`: ptr ONNXTensorElementDataType): OrtStatusPtr {.cdecl.}
    GetDimensionsCount*: proc (info: ptr OrtTensorTypeAndShapeInfo, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    GetDimensions*: proc (info: ptr OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize_t): OrtStatusPtr {.cdecl.}
    GetSymbolicDimensions*: proc (info: ptr OrtTensorTypeAndShapeInfo, dim_params: ptr cstring, dim_params_length: csize_t): OrtStatusPtr {.cdecl.}
    GetTensorShapeElementCount*: proc (info: ptr OrtTensorTypeAndShapeInfo, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    GetTensorTypeAndShape*: proc (value: ptr OrtValue, `out`: ptr ptr OrtTensorTypeAndShapeInfo): OrtStatusPtr {.cdecl.}
    GetTypeInfo*: proc (value: ptr OrtValue, `out`: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    GetValueType*: proc (value: ptr OrtValue, `out`: ptr ONNXType): OrtStatusPtr {.cdecl.}
    CreateMemoryInfo*: proc (name1: cstring, `type`: OrtAllocatorType, id1: cint,mem_type1: OrtMemType, `out`: ptr ptr OrtMemoryInfo): OrtStatusPtr {.cdecl.}
    CreateCpuMemoryInfo*: proc (`type`: OrtAllocatorType, mem_type1: OrtMemType, `out`: ptr ptr OrtMemoryInfo): OrtStatusPtr {.cdecl.}
    CompareMemoryInfo*: proc (info1: ptr OrtMemoryInfo, info2: ptr OrtMemoryInfo, `out`: ptr cint): OrtStatusPtr {.cdecl.}
    MemoryInfoGetName*: proc (`ptr`: ptr OrtMemoryInfo, `out`: ptr cstring): OrtStatusPtr {.cdecl.}
    MemoryInfoGetId*: proc (`ptr`: ptr OrtMemoryInfo, `out`: ptr cint): OrtStatusPtr {.cdecl.}
    MemoryInfoGetMemType*: proc (`ptr`: ptr OrtMemoryInfo, `out`: ptr OrtMemType): OrtStatusPtr {.cdecl.}
    MemoryInfoGetType*: proc (`ptr`: ptr OrtMemoryInfo, `out`: ptr OrtAllocatorType): OrtStatusPtr {.cdecl.}
    AllocatorAlloc*: proc (`ptr`: ptr OrtAllocator, size: csize_t, `out`: ptr pointer): OrtStatusPtr {.cdecl.}
    AllocatorFree*: proc (`ptr`: ptr OrtAllocator, p: pointer): OrtStatusPtr {.cdecl.}
    AllocatorGetInfo*: proc (`ptr`: ptr OrtAllocator, `out`: ptr ptr OrtMemoryInfo): OrtStatusPtr {.cdecl.}
    GetAllocatorWithDefaultOptions*: proc (`out`: ptr ptr OrtAllocator): OrtStatusPtr {.cdecl.}
    AddFreeDimensionOverride*: proc (options: ptr OrtSessionOptions, dim_denotation: cstring, dim_value: int64): OrtStatusPtr {.cdecl.}
    GetValue*: proc (value: ptr OrtValue, index: cint, allocator: ptr OrtAllocator, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    GetValueCount*: proc (value: ptr OrtValue, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    CreateValue*: proc (`in`: ptr ptr OrtValue, num_values: csize_t, value_type: ONNXType, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    CreateOpaqueValue*: proc (domain_name: cstring, type_name: cstring, data_container: pointer, data_container_size: csize_t, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    GetOpaqueValue*: proc (domain_name: cstring, type_name: cstring, `in`: ptr OrtValue, data_container: pointer, data_container_size: csize_t): OrtStatusPtr {.cdecl.}
    KernelInfoGetAttribute_float*: proc (info: ptr OrtKernelInfo, name: cstring, `out`: ptr cfloat): OrtStatusPtr {.cdecl.}
    KernelInfoGetAttribute_int64*: proc (info: ptr OrtKernelInfo, name: cstring, `out`: ptr int64): OrtStatusPtr {.cdecl.}
    KernelInfoGetAttribute_string*: proc (info: ptr OrtKernelInfo, name: cstring,  `out`: cstring, size: ptr csize_t): OrtStatusPtr {.cdecl.}
    KernelContext_GetInputCount*: proc (context: ptr OrtKernelContext, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    KernelContext_GetOutputCount*: proc (context: ptr OrtKernelContext, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    KernelContext_GetInput*: proc (context: ptr OrtKernelContext, index: csize_t, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    KernelContext_GetOutput*: proc (context: ptr OrtKernelContext, index: csize_t, dim_values: ptr int64, dim_count: csize_t, `out`: ptr ptr OrtValue): OrtStatusPtr {.cdecl.}
    ReleaseEnv*: proc (input: ptr OrtEnv) {.cdecl.}
    ReleaseStatus*: proc (input: ptr OrtStatus) {.cdecl.}
    ReleaseMemoryInfo*: proc (input: ptr OrtMemoryInfo) {.cdecl.}
    ReleaseSession*: proc (input: ptr OrtSession) {.cdecl.}
    ReleaseValue*: proc (input: ptr OrtValue) {.cdecl.}
    ReleaseRunOptions*: proc (input: ptr OrtRunOptions) {.cdecl.}
    ReleaseTypeInfo*: proc (input: ptr OrtTypeInfo) {.cdecl.}
    ReleaseTensorTypeAndShapeInfo*: proc (input: ptr OrtTensorTypeAndShapeInfo) {.cdecl.}
    ReleaseSessionOptions*: proc (input: ptr OrtSessionOptions) {.cdecl.}
    ReleaseCustomOpDomain*: proc (input: ptr OrtCustomOpDomain) {.cdecl.}
    GetDenotationFromTypeInfo*: proc (a1: ptr OrtTypeInfo, denotation: ptr cstring, len: ptr csize_t): OrtStatusPtr {.cdecl.}
    CastTypeInfoToMapTypeInfo*: proc (type_info: ptr OrtTypeInfo, `out`: ptr ptr OrtMapTypeInfo): OrtStatusPtr {.cdecl.}
    CastTypeInfoToSequenceTypeInfo*: proc (type_info: ptr OrtTypeInfo,`out`: ptr ptr OrtSequenceTypeInfo): OrtStatusPtr {.cdecl.}
    GetMapKeyType*: proc (map_type_info: ptr OrtMapTypeInfo, `out`: ptr ONNXTensorElementDataType): OrtStatusPtr {.cdecl.}
    GetMapValueType*: proc (map_type_info: ptr OrtMapTypeInfo, type_info: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    GetSequenceElementType*: proc (sequence_type_info: ptr OrtSequenceTypeInfo, type_info: ptr ptr OrtTypeInfo): OrtStatusPtr {.cdecl.}
    ReleaseMapTypeInfo*: proc (input: ptr OrtMapTypeInfo) {.cdecl.}
    ReleaseSequenceTypeInfo*: proc (input: ptr OrtSequenceTypeInfo) {.cdecl.}
    SessionEndProfiling*: proc (sess: ptr OrtSession, allocator: ptr OrtAllocator, `out`: ptr cstring): OrtStatusPtr {.cdecl.}
    SessionGetModelMetadata*: proc (sess: ptr OrtSession, `out`: ptr ptr OrtModelMetadata): OrtStatusPtr {.cdecl.}
    ModelMetadataGetProducerName*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    ModelMetadataGetGraphName*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    ModelMetadataGetDomain*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    ModelMetadataGetDescription*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): OrtStatusPtr {.cdecl.}
    ModelMetadataLookupCustomMetadataMap*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator,key: cstring, value: ptr cstring): OrtStatusPtr {.cdecl.}
    ModelMetadataGetVersion*: proc (model_metadata: ptr OrtModelMetadata, value: ptr int64): OrtStatusPtr {.cdecl.}
    ReleaseModelMetadata*: proc (input: ptr OrtModelMetadata) {.cdecl.}
    CreateEnvWithGlobalThreadPools*: proc (logging_level: OrtLoggingLevel,logid: cstring, t_options: ptr OrtThreadingOptions, `out`: ptr ptr OrtEnv): OrtStatusPtr {.cdecl.}
    DisablePerSessionThreads*: proc (options: ptr OrtSessionOptions): OrtStatusPtr {.cdecl.}
    CreateThreadingOptions*: proc (`out`: ptr ptr OrtThreadingOptions): OrtStatusPtr {.cdecl.}
    ReleaseThreadingOptions*: proc (input: ptr OrtThreadingOptions) {.cdecl.}
    ModelMetadataGetCustomMetadataMapKeys*: proc (model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator,keys: ptr ptr cstring, num_keys: ptr int64): OrtStatusPtr {.cdecl.}
    AddFreeDimensionOverrideByName*: proc (options: ptr OrtSessionOptions,dim_name: cstring, dim_value: int64): OrtStatusPtr {.cdecl.}
    GetAvailableProviders*: proc (out_ptr: ptr ptr cstring, provider_length: ptr cint): OrtStatusPtr {.cdecl.}
    ReleaseAvailableProviders*: proc (`ptr`: ptr cstring, providers_length: cint): OrtStatusPtr {.cdecl.}
    GetStringTensorElementLength*: proc (value: ptr OrtValue, index: csize_t, `out`: ptr csize_t): OrtStatusPtr {.cdecl.}
    GetStringTensorElement*: proc (value: ptr OrtValue, s_len: csize_t, index: csize_t, s: pointer): OrtStatusPtr {.cdecl.}
    FillStringTensorElement*: proc (value: ptr OrtValue, s: cstring, index: csize_t): OrtStatusPtr {.cdecl.}
    AddSessionConfigEntry*: proc (options: ptr OrtSessionOptions, config_key: cstring, config_value: cstring): OrtStatusPtr {.cdecl.}
    CreateAllocator*: proc (sess: ptr OrtSession, mem_info: ptr OrtMemoryInfo, `out`: ptr ptr OrtAllocator): OrtStatusPtr {.cdecl.}
    ReleaseAllocator*: proc (input: ptr OrtAllocator) {.cdecl.}
    RunWithBinding*: proc (sess: ptr OrtSession, run_options: ptr OrtRunOptions, binding_ptr: ptr OrtIoBinding): OrtStatusPtr {.cdecl.}
    CreateIoBinding*: proc (sess: ptr OrtSession, `out`: ptr ptr OrtIoBinding): OrtStatusPtr {.cdecl.}
    ReleaseIoBinding*: proc (input: ptr OrtIoBinding) {.cdecl.}
    BindInput*: proc (binding_ptr: ptr OrtIoBinding, name: cstring,val_ptr: ptr OrtValue): OrtStatusPtr {.cdecl.}
    BindOutput*: proc (binding_ptr: ptr OrtIoBinding, name: cstring, val_ptr: ptr OrtValue): OrtStatusPtr {.cdecl.}
    BindOutputToDevice*: proc (binding_ptr: ptr OrtIoBinding, name: cstring, val_ptr: ptr OrtMemoryInfo): OrtStatusPtr {.cdecl.}
    GetBoundOutputNames*: proc (binding_ptr: ptr OrtIoBinding, allocator: ptr OrtAllocator, buffer: ptr cstring, lengths: ptr ptr csize_t, count: ptr csize_t): OrtStatusPtr {.cdecl.}
    GetBoundOutputValues*: proc (binding_ptr: ptr OrtIoBinding, allocator: ptr OrtAllocator, output: ptr ptr ptr OrtValue, output_count: ptr csize_t): OrtStatusPtr {.cdecl.}
    ClearBoundInputs*: proc (binding_ptr: ptr OrtIoBinding) {.cdecl.}
    ClearBoundOutputs*: proc (binding_ptr: ptr OrtIoBinding) {.cdecl.}
    TensorAt*: proc (value: ptr OrtValue, location_values: ptr int64, location_values_count: csize_t, `out`: ptr pointer): OrtStatusPtr {.cdecl.}
    CreateAndRegisterAllocator*: proc (env: ptr OrtEnv, mem_info: ptr OrtMemoryInfo, arena_cfg: ptr OrtArenaCfg): OrtStatusPtr {.cdecl.}
    SetLanguageProjection*: proc (ort_env: ptr OrtEnv, projection: OrtLanguageProjection): OrtStatusPtr {.cdecl.}
    SessionGetProfilingStartTimeNs*: proc (sess: ptr OrtSession, `out`: ptr uint64): OrtStatusPtr {.cdecl.}
    SetGlobalIntraOpNumThreads*: proc (tp_options: ptr OrtThreadingOptions, intra_op_num_threads: cint): OrtStatusPtr {.cdecl.}
    SetGlobalInterOpNumThreads*: proc (tp_options: ptr OrtThreadingOptions, inter_op_num_threads: cint): OrtStatusPtr {.cdecl.}
    SetGlobalSpinControl*: proc (tp_options: ptr OrtThreadingOptions, allow_spinning: cint): OrtStatusPtr {.cdecl.}
    AddInitializer*: proc (options: ptr OrtSessionOptions, name: cstring, val: ptr OrtValue): OrtStatusPtr {.cdecl.}
    CreateEnvWithCustomLoggerAndGlobalThreadPools*: proc (logging_function: OrtLoggingFunction, logger_param: pointer,logging_level: OrtLoggingLevel, logid: cstring,tp_options: ptr OrtThreadingOptions, `out`: ptr ptr OrtEnv): OrtStatusPtr {.cdecl.}
    SessionOptionsAppendExecutionProvider_CUDA*: proc (options: ptr OrtSessionOptions, cuda_options: ptr OrtCUDAProviderOptions): OrtStatusPtr {.cdecl.}
    SessionOptionsAppendExecutionProvider_OpenVINO*: proc (options: ptr OrtSessionOptions,provider_options: ptr OrtOpenVINOProviderOptions): OrtStatusPtr {.cdecl.}
    SetGlobalDenormalAsZero*: proc (tp_options: ptr OrtThreadingOptions): OrtStatusPtr {.cdecl.}
    CreateArenaCfg*: proc (max_mem: csize_t, arena_extend_strategy: cint, initial_chunk_size_bytes: cint, max_dead_bytes_per_chunk: cint, `out`: ptr ptr OrtArenaCfg): OrtStatusPtr {.cdecl.}
    ReleaseArenaCfg*: proc (input: ptr OrtArenaCfg) {.cdecl.}

  OrtCustomOp* {.bycopy.} = object
    version*: uint32
    CreateKernel*: proc (op: ptr OrtCustomOp, api: ptr OrtApi, info: ptr OrtKernelInfo): pointer {.cdecl.}
    GetName*: proc (op: ptr OrtCustomOp): cstring {.cdecl.}
    GetExecutionProviderType*: proc (op: ptr OrtCustomOp): cstring {.cdecl.}
    GetInputType*: proc (op: ptr OrtCustomOp, index: csize_t): ONNXTensorElementDataType {.cdecl.}
    GetInputTypeCount*: proc (op: ptr OrtCustomOp): csize_t {.cdecl.}
    GetOutputType*: proc (op: ptr OrtCustomOp, index: csize_t): ONNXTensorElementDataType {.cdecl.}
    GetOutputTypeCount*: proc (op: ptr OrtCustomOp): csize_t {.cdecl.}
    KernelCompute*: proc (op_kernel: pointer, context: ptr OrtKernelContext) {.cdecl.}
    KernelDestroy*: proc (op_kernel: pointer) {.cdecl.}

type
  OrtApiBase* {.bycopy.} = object
    GetApi*: proc (version: uint32): ptr OrtApi {.cdecl.}
    GetVersionString*: proc (): cstring {.cdecl.}

proc OrtGetApiBase*(): ptr OrtApiBase {.cdecl, importc: "OrtGetApiBase", dynlib: libname.}