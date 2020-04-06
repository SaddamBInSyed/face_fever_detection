import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import os

model_path ="models/mxnet_exported_R50.onnx" #"models/resnet18v1.onnx"# "models/resnet18v1.onnx" #"models/mxnet_exported_R50.onnx"
input_size = 32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

def build_engine(onnx_file_path, verbose=False):
    """Takes an ONNX file and creates a TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True
        #builder.strict_type_constraints = True

        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        if trt.__version__[0] >= '7':
            # The actual yolov3.onnx is generated with batch size 64.
            # Reshape input to batch size 1
            shape = list(network.get_input(0).shape)
            shape[0] = 1
            network.get_input(0).shape = shape
        print('Completed parsing of ONNX file')

        print('Building an engine; this may take a while...')
        engine = builder.build_cuda_engine(network)
        print('Completed creating engine')
        # with open(engine_file_path, 'wb') as f:
        #     f.write(engine.serialize())
        return engine

# def build_engine(model_path):
#     with trt.Builder(TRT_LOGGER) as builder, \
#         builder.create_network() as network, \
#         trt.OnnxParser(network, TRT_LOGGER) as parser:
#             builder.max_workspace_size = 1<<20
#             builder.max_batch_size = 1
#             with open(model_path, "rb") as f:
#                 parser.parse(f.read())
#             engine = builder.build_cuda_engine(network)
#             return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

if __name__ == "__main__":
    inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
    engine = build_engine(model_path, verbose=True)
    import pdb
    pdb.set_trace()
    context = engine.create_execution_context()
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)
        print("cost time: ", time.time()-t1)


# tensorrt docker image: docker pull nvcr.io/nvidia/tensorrt:19.09-py3 (See: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags)
# NOTE: cuda driver >= 418