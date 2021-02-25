#ifdef C2NIM
# dynlib libname
# cdecl
# define libname "/mnt/c/Users/weixi/Downloads/onnxruntime-win-x64-gpu-1.6.0/lib/onnxruntime.dll"
#endif

typedef enum ONNXTensorElementDataType
{
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
}

ONNXTensorElementDataType;
typedef enum ONNXType
{
	ONNX_TYPE_UNKNOWN,
	ONNX_TYPE_TENSOR,
	ONNX_TYPE_SEQUENCE,
	ONNX_TYPE_MAP,
	ONNX_TYPE_OPAQUE,
	ONNX_TYPE_SPARSETENSOR,
}

ONNXType;
typedef enum OrtLoggingLevel
{
	ORT_LOGGING_LEVEL_VERBOSE,
	ORT_LOGGING_LEVEL_INFO,
	ORT_LOGGING_LEVEL_WARNING,
	ORT_LOGGING_LEVEL_ERROR,
	ORT_LOGGING_LEVEL_FATAL,
}

OrtLoggingLevel;
typedef enum OrtErrorCode
{
	ORT_OK,
	ORT_FAIL,
	ORT_INVALID_ARGUMENT,
	ORT_NO_SUCHFILE,
	ORT_NO_MODEL,
	ORT_ENGINE_ERROR,
	ORT_RUNTIME_EXCEPTION,
	ORT_INVALID_PROTOBUF,
	ORT_MODEL_LOADED,
	ORT_NOT_IMPLEMENTED,
	ORT_INVALID_GRAPH,
	ORT_EP_FAIL,
}

OrtErrorCode;
struct OrtEnv;
typedef struct OrtEnv OrtEnv;;
struct OrtStatus;
typedef struct OrtStatus OrtStatus;;
struct OrtMemoryInfo;
typedef struct OrtMemoryInfo OrtMemoryInfo;;
struct OrtIoBinding;
typedef struct OrtIoBinding OrtIoBinding;;
struct OrtSession;
typedef struct OrtSession OrtSession;;
struct OrtValue;
typedef struct OrtValue OrtValue;;
struct OrtRunOptions;
typedef struct OrtRunOptions OrtRunOptions;;
struct OrtTypeInfo;
typedef struct OrtTypeInfo OrtTypeInfo;;
struct OrtTensorTypeAndShapeInfo;
typedef struct OrtTensorTypeAndShapeInfo OrtTensorTypeAndShapeInfo;;
struct OrtSessionOptions;
typedef struct OrtSessionOptions OrtSessionOptions;;
struct OrtCustomOpDomain;
typedef struct OrtCustomOpDomain OrtCustomOpDomain;;
struct OrtMapTypeInfo;
typedef struct OrtMapTypeInfo OrtMapTypeInfo;;
struct OrtSequenceTypeInfo;
typedef struct OrtSequenceTypeInfo OrtSequenceTypeInfo;;
struct OrtModelMetadata;
typedef struct OrtModelMetadata OrtModelMetadata;;
struct OrtThreadPoolParams;
typedef struct OrtThreadPoolParams OrtThreadPoolParams;;
struct OrtThreadingOptions;
typedef struct OrtThreadingOptions OrtThreadingOptions;;
struct OrtArenaCfg;
typedef struct OrtArenaCfg OrtArenaCfg;;
typedef OrtStatus * OrtStatusPtr;
typedef struct OrtAllocator
{
	uint32_t version;
	void *(*Alloc)(struct OrtAllocator *this, size_t size);
	void(*Free)(struct OrtAllocator *this, void *p);
	const struct OrtMemoryInfo * (*Info)(const struct OrtAllocator *this);
}

OrtAllocator;
typedef void(*OrtLoggingFunction)(
	void *param, OrtLoggingLevel severity, const char *category, const char *logid, const char *code_location, const char *message);
typedef enum GraphOptimizationLevel
{
	ORT_DISABLE_ALL = 0,
		ORT_ENABLE_BASIC = 1,
		ORT_ENABLE_EXTENDED = 2,
		ORT_ENABLE_ALL = 99
}

GraphOptimizationLevel;
typedef enum ExecutionMode
{
	ORT_SEQUENTIAL = 0,
		ORT_PARALLEL = 1,
}

ExecutionMode;
typedef enum OrtLanguageProjection
{
	ORT_PROJECTION_C = 0,
		ORT_PROJECTION_CPLUSPLUS = 1,
		ORT_PROJECTION_CSHARP = 2,
		ORT_PROJECTION_PYTHON = 3,
		ORT_PROJECTION_JAVA = 4,
		ORT_PROJECTION_WINML = 5,
		ORT_PROJECTION_NODEJS = 6,
}

OrtLanguageProjection;
struct OrtKernelInfo;
typedef struct OrtKernelInfo OrtKernelInfo;
struct OrtKernelContext;
typedef struct OrtKernelContext OrtKernelContext;
struct OrtCustomOp;
typedef struct OrtCustomOp OrtCustomOp;
typedef enum OrtAllocatorType
{
	Invalid = -1,
		OrtDeviceAllocator = 0,
		OrtArenaAllocator = 1
}

OrtAllocatorType;
typedef enum OrtMemType
{
	OrtMemTypeCPUInput = -2,
		OrtMemTypeCPUOutput = -1,
		OrtMemTypeCPU = OrtMemTypeCPUOutput,
		OrtMemTypeDefault = 0,
}

OrtMemType;
typedef enum OrtCudnnConvAlgoSearch
{
	EXHAUSTIVE,
	HEURISTIC,
	DEFAULT,
}

OrtCudnnConvAlgoSearch;
typedef struct OrtCUDAProviderOptions
{
	int device_id;
	OrtCudnnConvAlgoSearch cudnn_conv_algo_search;
	size_t cuda_mem_limit;
	int arena_extend_strategy;
	int do_copy_in_default_stream;
}

OrtCUDAProviderOptions;
typedef struct OrtOpenVINOProviderOptions
{
	const char *device_type;
	unsigned char enable_vpu_fast_compile;
	const char *device_id;
	size_t num_of_threads;
}

OrtOpenVINOProviderOptions;
struct OrtApi;
typedef struct OrtApi OrtApi;
struct OrtApiBase
{
	const OrtApi * (*GetApi)(uint32_t version);
	const char *(*GetVersionString)();
};
typedef struct OrtApiBase OrtApiBase;
const OrtApiBase* OrtGetApiBase(void);
struct OrtApi
{
	OrtStatus * (*CreateStatus)(OrtErrorCode code, const char *msg);
	OrtErrorCode(*GetErrorCode)(const OrtStatus *status);
	const char *(*GetErrorMessage)(const OrtStatus *status);
	OrtStatusPtr(*CreateEnv)(OrtLoggingLevel logging_level, const char *logid, OrtEnv **out);
	OrtStatusPtr(*CreateEnvWithCustomLogger)(OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel logging_level, const char *logid, OrtEnv **out);
	OrtStatusPtr(*EnableTelemetryEvents)(const OrtEnv *env);
	OrtStatusPtr(*DisableTelemetryEvents)(const OrtEnv *env);
	OrtStatusPtr(*CreateSession)(const OrtEnv *env, const char *model_path, const OrtSessionOptions *options, OrtSession **out);
	OrtStatusPtr(*CreateSessionFromArray)(const OrtEnv *env, const void *model_data, size_t model_data_length, const OrtSessionOptions *options, OrtSession **out);
	OrtStatusPtr(*Run)(OrtSession *sess, const OrtRunOptions *run_options, const char *const *input_names, const OrtValue *const *input, size_t input_len, const char *const *output_names1, size_t output_names_len, OrtValue **output);
	OrtStatusPtr(*CreateSessionOptions)(OrtSessionOptions **options);
	OrtStatusPtr(*SetOptimizedModelFilePath)(OrtSessionOptions *options, const char *optimized_model_filepath);
	OrtStatusPtr(*CloneSessionOptions)(const OrtSessionOptions *in_options, OrtSessionOptions **out_options);
	OrtStatusPtr(*SetSessionExecutionMode)(OrtSessionOptions *options, ExecutionMode execution_mode);
	OrtStatusPtr(*EnableProfiling)(OrtSessionOptions *options, const char *profile_file_prefix);
	OrtStatusPtr(*DisableProfiling)(OrtSessionOptions *options);
	OrtStatusPtr(*EnableMemPattern)(OrtSessionOptions *options);
	OrtStatusPtr(*DisableMemPattern)(OrtSessionOptions *options);
	OrtStatusPtr(*EnableCpuMemArena)(OrtSessionOptions *options);
	OrtStatusPtr(*DisableCpuMemArena)(OrtSessionOptions *options);
	OrtStatusPtr(*SetSessionLogId)(OrtSessionOptions *options, const char *logid);
	OrtStatusPtr(*SetSessionLogVerbosityLevel)(OrtSessionOptions *options, int session_log_verbosity_level);
	OrtStatusPtr(*SetSessionLogSeverityLevel)(OrtSessionOptions *options, int session_log_severity_level);
	OrtStatusPtr(*SetSessionGraphOptimizationLevel)(OrtSessionOptions *options, GraphOptimizationLevel graph_optimization_level);
	OrtStatusPtr(*SetIntraOpNumThreads)(OrtSessionOptions *options, int intra_op_num_threads);
	OrtStatusPtr(*SetInterOpNumThreads)(OrtSessionOptions *options, int inter_op_num_threads);
	OrtStatusPtr(*CreateCustomOpDomain)(const char *domain, OrtCustomOpDomain **out);
	OrtStatusPtr(*CustomOpDomain_Add)(OrtCustomOpDomain *custom_op_domain, const OrtCustomOp *op);
	OrtStatusPtr(*AddCustomOpDomain)(OrtSessionOptions *options, OrtCustomOpDomain *custom_op_domain);
	OrtStatusPtr(*RegisterCustomOpsLibrary)(OrtSessionOptions *options, const char *library_path, void **library_handle);
	OrtStatusPtr(*SessionGetInputCount)(const OrtSession *sess, size_t *out);
	OrtStatusPtr(*SessionGetOutputCount)(const OrtSession *sess, size_t *out);
	OrtStatusPtr(*SessionGetOverridableInitializerCount)(const OrtSession *sess, size_t *out);
	OrtStatusPtr(*SessionGetInputTypeInfo)(const OrtSession *sess, size_t index, OrtTypeInfo **type_info);
	OrtStatusPtr(*SessionGetOutputTypeInfo)(const OrtSession *sess, size_t index, OrtTypeInfo **type_info);
	OrtStatusPtr(*SessionGetOverridableInitializerTypeInfo)(const OrtSession *sess, size_t index, OrtTypeInfo **type_info);
	OrtStatusPtr(*SessionGetInputName)(const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*SessionGetOutputName)(const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*SessionGetOverridableInitializerName)(const OrtSession *sess, size_t index, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*CreateRunOptions)(OrtRunOptions **out);
	OrtStatusPtr(*RunOptionsSetRunLogVerbosityLevel)(OrtRunOptions *options, int value);
	OrtStatusPtr(*RunOptionsSetRunLogSeverityLevel)(OrtRunOptions *options, int value);
	OrtStatusPtr(*RunOptionsSetRunTag)(OrtRunOptions *, const char *run_tag);
	OrtStatusPtr(*RunOptionsGetRunLogVerbosityLevel)(const OrtRunOptions *options, int *out);
	OrtStatusPtr(*RunOptionsGetRunLogSeverityLevel)(const OrtRunOptions *options, int *out);
	OrtStatusPtr(*RunOptionsGetRunTag)(const OrtRunOptions *, const char **out);
	OrtStatusPtr(*RunOptionsSetTerminate)(OrtRunOptions *options);
	OrtStatusPtr(*RunOptionsUnsetTerminate)(OrtRunOptions *options);
	OrtStatusPtr(*CreateTensorAsOrtValue)(OrtAllocator *allocator, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out);
	OrtStatusPtr(*CreateTensorWithDataAsOrtValue)(const OrtMemoryInfo *info, void *p_data, size_t p_data_len, const int64_t *shape, size_t shape_len, ONNXTensorElementDataType type, OrtValue **out);
	OrtStatusPtr(*IsTensor)(const OrtValue *value, int *out);
	OrtStatusPtr(*GetTensorMutableData)(OrtValue *value, void **out);
	OrtStatusPtr(*FillStringTensor)(OrtValue *value, const char *const *s, size_t s_len);
	OrtStatusPtr(*GetStringTensorDataLength)(const OrtValue *value, size_t *len);
	OrtStatusPtr(*GetStringTensorContent)(const OrtValue *value, void *s, size_t s_len, size_t *offsets, size_t offsets_len);
	OrtStatusPtr(*CastTypeInfoToTensorInfo)(const OrtTypeInfo *, const OrtTensorTypeAndShapeInfo **out);
	OrtStatusPtr(*GetOnnxTypeFromTypeInfo)(const OrtTypeInfo *, enum ONNXType *out);
	OrtStatusPtr(*CreateTensorTypeAndShapeInfo)(OrtTensorTypeAndShapeInfo **out);
	OrtStatusPtr(*SetTensorElementType)(OrtTensorTypeAndShapeInfo *, enum ONNXTensorElementDataType type);
	OrtStatusPtr(*SetDimensions)(OrtTensorTypeAndShapeInfo *info, const int64_t *dim_values, size_t dim_count);
	OrtStatusPtr(*GetTensorElementType)(const OrtTensorTypeAndShapeInfo *, enum ONNXTensorElementDataType *out);
	OrtStatusPtr(*GetDimensionsCount)(const OrtTensorTypeAndShapeInfo *info, size_t *out);
	OrtStatusPtr(*GetDimensions)(const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);
	OrtStatusPtr(*GetSymbolicDimensions)(const OrtTensorTypeAndShapeInfo *info, const char *dim_params[], size_t dim_params_length);
	OrtStatusPtr(*GetTensorShapeElementCount)(const OrtTensorTypeAndShapeInfo *info, size_t *out);
	OrtStatusPtr(*GetTensorTypeAndShape)(const OrtValue *value, OrtTensorTypeAndShapeInfo **out);
	OrtStatusPtr(*GetTypeInfo)(const OrtValue *value, OrtTypeInfo **out);
	OrtStatusPtr(*GetValueType)(const OrtValue *value, enum ONNXType *out);
	OrtStatusPtr(*CreateMemoryInfo)(const char *name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, OrtMemoryInfo **out);
	OrtStatusPtr(*CreateCpuMemoryInfo)(enum OrtAllocatorType type, enum OrtMemType mem_type1, OrtMemoryInfo **out);
	OrtStatusPtr(*CompareMemoryInfo)(const OrtMemoryInfo *info1, const OrtMemoryInfo *info2, int *out);
	OrtStatusPtr(*MemoryInfoGetName)(const OrtMemoryInfo *ptr, const char **out);
	OrtStatusPtr(*MemoryInfoGetId)(const OrtMemoryInfo *ptr, int *out);
	OrtStatusPtr(*MemoryInfoGetMemType)(const OrtMemoryInfo *ptr, OrtMemType *out);
	OrtStatusPtr(*MemoryInfoGetType)(const OrtMemoryInfo *ptr, OrtAllocatorType *out);
	OrtStatusPtr(*AllocatorAlloc)(OrtAllocator *ptr, size_t size, void **out);
	OrtStatusPtr(*AllocatorFree)(OrtAllocator *ptr, void *p);
	OrtStatusPtr(*AllocatorGetInfo)(const OrtAllocator *ptr, const struct OrtMemoryInfo **out);
	OrtStatusPtr(*GetAllocatorWithDefaultOptions)(OrtAllocator **out);
	OrtStatusPtr(*AddFreeDimensionOverride)(OrtSessionOptions *options, const char *dim_denotation, int64_t dim_value);
	OrtStatusPtr(*GetValue)(const OrtValue *value, int index, OrtAllocator *allocator, OrtValue **out);
	OrtStatusPtr(*GetValueCount)(const OrtValue *value, size_t *out);
	OrtStatusPtr(*CreateValue)(const OrtValue *const *in, size_t num_values, enum ONNXType value_type, OrtValue **out);
	OrtStatusPtr(*CreateOpaqueValue)(const char *domain_name, const char *type_name, const void *data_container, size_t data_container_size, OrtValue **out);
	OrtStatusPtr(*GetOpaqueValue)(const char *domain_name, const char *type_name, const OrtValue *in, void *data_container, size_t data_container_size);
	OrtStatusPtr(*KernelInfoGetAttribute_float)(const OrtKernelInfo *info, const char *name, float *out);
	OrtStatusPtr(*KernelInfoGetAttribute_int64)(const OrtKernelInfo *info, const char *name, int64_t *out);
	OrtStatusPtr(*KernelInfoGetAttribute_string)(const OrtKernelInfo *info, const char *name, char *out, size_t *size);
	OrtStatusPtr(*KernelContext_GetInputCount)(const OrtKernelContext *context, size_t *out);
	OrtStatusPtr(*KernelContext_GetOutputCount)(const OrtKernelContext *context, size_t *out);
	OrtStatusPtr(*KernelContext_GetInput)(const OrtKernelContext *context, size_t index, const OrtValue **out);
	OrtStatusPtr(*KernelContext_GetOutput)(OrtKernelContext *context, size_t index, const int64_t *dim_values, size_t dim_count, OrtValue **out);
	void(*ReleaseEnv)(OrtEnv *input);
	void(*ReleaseStatus)(OrtStatus *input);
	void(*ReleaseMemoryInfo)(OrtMemoryInfo *input);
	void(*ReleaseSession)(OrtSession *input);
	void(*ReleaseValue)(OrtValue *input);
	void(*ReleaseRunOptions)(OrtRunOptions *input);
	void(*ReleaseTypeInfo)(OrtTypeInfo *input);
	void(*ReleaseTensorTypeAndShapeInfo)(OrtTensorTypeAndShapeInfo *input);
	void(*ReleaseSessionOptions)(OrtSessionOptions *input);
	void(*ReleaseCustomOpDomain)(OrtCustomOpDomain *input);
	OrtStatusPtr(*GetDenotationFromTypeInfo)(const OrtTypeInfo *, const char **const denotation, size_t *len);
	OrtStatusPtr(*CastTypeInfoToMapTypeInfo)(const OrtTypeInfo *type_info, const OrtMapTypeInfo **out);
	OrtStatusPtr(*CastTypeInfoToSequenceTypeInfo)(const OrtTypeInfo *type_info, const OrtSequenceTypeInfo **out);
	OrtStatusPtr(*GetMapKeyType)(const OrtMapTypeInfo *map_type_info, enum ONNXTensorElementDataType *out);
	OrtStatusPtr(*GetMapValueType)(const OrtMapTypeInfo *map_type_info, OrtTypeInfo **type_info);
	OrtStatusPtr(*GetSequenceElementType)(const OrtSequenceTypeInfo *sequence_type_info, OrtTypeInfo **type_info);
	void(*ReleaseMapTypeInfo)(OrtMapTypeInfo *input);
	void(*ReleaseSequenceTypeInfo)(OrtSequenceTypeInfo *input);
	OrtStatusPtr(*SessionEndProfiling)(OrtSession *sess, OrtAllocator *allocator, char **out);
	OrtStatusPtr(*SessionGetModelMetadata)(const OrtSession *sess, OrtModelMetadata **out);
	OrtStatusPtr(*ModelMetadataGetProducerName)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*ModelMetadataGetGraphName)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*ModelMetadataGetDomain)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*ModelMetadataGetDescription)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char **value);
	OrtStatusPtr(*ModelMetadataLookupCustomMetadataMap)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, const char *key, char **value);
	OrtStatusPtr(*ModelMetadataGetVersion)(const OrtModelMetadata *model_metadata, int64_t *value);
	void(*ReleaseModelMetadata)(OrtModelMetadata *input);
	OrtStatusPtr(*CreateEnvWithGlobalThreadPools)(OrtLoggingLevel logging_level, const char *logid, const OrtThreadingOptions *t_options, OrtEnv **out);
	OrtStatusPtr(*DisablePerSessionThreads)(OrtSessionOptions *options);
	OrtStatusPtr(*CreateThreadingOptions)(OrtThreadingOptions **out);
	void(*ReleaseThreadingOptions)(OrtThreadingOptions *input);
	OrtStatusPtr(*ModelMetadataGetCustomMetadataMapKeys)(const OrtModelMetadata *model_metadata, OrtAllocator *allocator, char ***keys, int64_t *num_keys);
	OrtStatusPtr(*AddFreeDimensionOverrideByName)(OrtSessionOptions *options, const char *dim_name, int64_t dim_value);
	OrtStatusPtr(*GetAvailableProviders)(char ***out_ptr, int *provider_length);
	OrtStatusPtr(*ReleaseAvailableProviders)(char **ptr, int providers_length);
	OrtStatusPtr(*GetStringTensorElementLength)(const OrtValue *value, size_t index, size_t *out);
	OrtStatusPtr(*GetStringTensorElement)(const OrtValue *value, size_t s_len, size_t index, void *s);
	OrtStatusPtr(*FillStringTensorElement)(OrtValue *value, const char *s, size_t index);
	OrtStatusPtr(*AddSessionConfigEntry)(OrtSessionOptions *options, const char *config_key, const char *config_value);
	OrtStatusPtr(*CreateAllocator)(const OrtSession *sess, const OrtMemoryInfo *mem_info, OrtAllocator **out);
	void(*ReleaseAllocator)(OrtAllocator *input);
	OrtStatusPtr(*RunWithBinding)(OrtSession *sess, const OrtRunOptions *run_options, const OrtIoBinding *binding_ptr);
	OrtStatusPtr(*CreateIoBinding)(OrtSession *sess, OrtIoBinding **out);
	void(*ReleaseIoBinding)(OrtIoBinding *input);
	OrtStatusPtr(*BindInput)(OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr);
	OrtStatusPtr(*BindOutput)(OrtIoBinding *binding_ptr, const char *name, const OrtValue *val_ptr);
	OrtStatusPtr(*BindOutputToDevice)(OrtIoBinding *binding_ptr, const char *name, const OrtMemoryInfo *val_ptr);
	OrtStatusPtr(*GetBoundOutputNames)(const OrtIoBinding *binding_ptr, OrtAllocator *allocator, char **buffer, size_t **lengths, size_t *count);
	OrtStatusPtr(*GetBoundOutputValues)(const OrtIoBinding *binding_ptr, OrtAllocator *allocator, OrtValue ***output, size_t *output_count);
	void(*ClearBoundInputs)(OrtIoBinding *binding_ptr);
	void(*ClearBoundOutputs)(OrtIoBinding *binding_ptr);
	OrtStatusPtr(*TensorAt)(OrtValue *value, const int64_t *location_values, size_t location_values_count, void **out);
	OrtStatusPtr(*CreateAndRegisterAllocator)(OrtEnv *env, const OrtMemoryInfo *mem_info, const OrtArenaCfg *arena_cfg);
	OrtStatusPtr(*SetLanguageProjection)(const OrtEnv *ort_env, OrtLanguageProjection projection);
	OrtStatusPtr(*SessionGetProfilingStartTimeNs)(const OrtSession *sess, uint64_t *out);
	OrtStatusPtr(*SetGlobalIntraOpNumThreads)(OrtThreadingOptions *tp_options, int intra_op_num_threads);
	OrtStatusPtr(*SetGlobalInterOpNumThreads)(OrtThreadingOptions *tp_options, int inter_op_num_threads);
	OrtStatusPtr(*SetGlobalSpinControl)(OrtThreadingOptions *tp_options, int allow_spinning);
	OrtStatusPtr(*AddInitializer)(OrtSessionOptions *options, const char *name, const OrtValue *val);
	OrtStatusPtr(*CreateEnvWithCustomLoggerAndGlobalThreadPools)(OrtLoggingFunction logging_function, void *logger_param, OrtLoggingLevel logging_level, const char *logid, const struct OrtThreadingOptions *tp_options, OrtEnv **out);
	OrtStatusPtr(*SessionOptionsAppendExecutionProvider_CUDA)(OrtSessionOptions *options, const OrtCUDAProviderOptions *cuda_options);
	OrtStatusPtr(*SessionOptionsAppendExecutionProvider_OpenVINO)(OrtSessionOptions *options, const OrtOpenVINOProviderOptions *provider_options);
	OrtStatusPtr(*SetGlobalDenormalAsZero)(OrtThreadingOptions *tp_options);
	OrtStatusPtr(*CreateArenaCfg)(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk, OrtArenaCfg **out);
	void(*ReleaseArenaCfg)(OrtArenaCfg *input);
};
struct OrtCustomOp
{
	uint32_t version;
	void *(*CreateKernel)(const struct OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info);
	const char *(*GetName)(const struct OrtCustomOp *op);
	const char *(*GetExecutionProviderType)(const struct OrtCustomOp *op);
	ONNXTensorElementDataType(*GetInputType)(const struct OrtCustomOp *op, size_t index);
	size_t(*GetInputTypeCount)(const struct OrtCustomOp *op);
	ONNXTensorElementDataType(*GetOutputType)(const struct OrtCustomOp *op, size_t index);
	size_t(*GetOutputTypeCount)(const struct OrtCustomOp *op);
	void(*KernelCompute)(void *op_kernel, OrtKernelContext *context);
	void(*KernelDestroy)(void *op_kernel);
};