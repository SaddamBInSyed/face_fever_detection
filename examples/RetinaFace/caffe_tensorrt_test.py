import tensorrt as trt
# from tensorrt import parsers

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import os



INPUT_LAYERS = ['data']
OUTPUT_LAYERS = ['face_rpn_cls_prob_stride8','face_rpn_cls_prob_stride16','face_rpn_cls_prob_stride32']
OUTPUT_LAYERS = ['face_rpn_cls_score_stride8','face_rpn_cls_score_stride16','face_rpn_cls_score_stride32']

# NO_RESHAPE_SOFTMAX
OUTPUT_LAYERS  =['face_rpn_cls_score_stride8', 'face_rpn_bbox_pred_stride8', 'face_rpn_landmark_pred_stride8',
                 'face_rpn_cls_score_stride16', 'face_rpn_bbox_pred_stride16', 'face_rpn_landmark_pred_stride16',
                 'face_rpn_cls_score_stride32', 'face_rpn_bbox_pred_stride32', 'face_rpn_landmark_pred_stride32']

# WITH_RESHAPE_SOFTMAX
OUTPUT_LAYERS  =['face_rpn_cls_score_stride32', 'face_rpn_bbox_pred_stride32', 'face_rpn_landmark_pred_stride32',
                 'face_rpn_cls_score_stride16', 'face_rpn_bbox_pred_stride16', 'face_rpn_landmark_pred_stride16',
                 'face_rpn_cls_score_stride8', 'face_rpn_bbox_pred_stride8', 'face_rpn_landmark_pred_stride8'
                 ]



INPUT_H = 384 #288
INPUT_W =  288 #384
OUTPUT_SIZE = 9

MODEL_PROTOTXT = 'models/R50_SOFTMAX_VERT.prototxt'
CAFFE_MODEL = 'models/R50-0000.caffemodel'

# engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
#                                        MODEL_PROTOTXT,
#                                        CAFFE_MODEL,
#                                        1,
#                                        1 << 20,
#                                        OUTPUT_LAYERS,
#                                        trt.infer.DataType.FLOAT)
#
#
# runtime = trt.infer.create_infer_runtime(G_LOGGER)
# context = engine.create_execution_context()



# DATA = './data/mnist/'
# IMAGE_MEAN = './data/mnist/mnist_mean.binaryproto'

model_path ="models/mxnet_exported_R50.onnx" #"models/resnet18v1.onnx"# "models/resnet18v1.onnx" #"models/mxnet_exported_R50.onnx"
# input_size = 640

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

def build_engine(deploy_file, model_file, verbose=False):
    """Takes an ONNX file and creates a TensorRT engine."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True
        datatype =  trt.float32
        #builder.strict_type_constraints = True

        # Parse model file
        # if not os.path.exists(onnx_file_path):
        #     print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
        #     exit(0)
        # print('Loading ONNX file from path {}...'.format(onnx_file_path))
        # with open(onnx_file_path, 'rb') as model:
        #     print('Beginning ONNX file parsing')
        if not parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=datatype):
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
        print('Completed parsing of Caffe file')

        net_out_tmp = [network.get_layer(ln).get_output(0) for ln in range(network.num_layers) if
                   network.get_layer(ln).get_output(0).name in OUTPUT_LAYERS]


        assert len(net_out_tmp) == len(OUTPUT_LAYERS)
        net_out = [None] * len(OUTPUT_LAYERS)
        for nn in net_out_tmp:
            net_out[OUTPUT_LAYERS.index(nn.name)]=nn

        assert None not in net_out
        for nn in net_out:
            network.mark_output(nn)

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
class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def alloc_buf(engine):
    # host cpu mem
    inputs = []
    outputs = [None]*OUTPUT_SIZE
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs[OUTPUT_LAYERS.index(binding)] = HostDeviceMem(host_mem, device_mem)
            # outputs.append(HostDeviceMem(host_mem, device_mem))



    return inputs, outputs, bindings, stream

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

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)
    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    #     cuda.memcpy_htod_async(in_gpu, inputs, stream)
    #     context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    #     cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    #     stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

def _preprocess_img(img, shape):
    """Preprocess an image before TRT YOLOv3 inferencing."""
    # img = cv2.resize(img, shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img


if __name__ == "__main__":
    # inputs = np.random.random((1, 3, INPUT_W, INPUT_H)).astype(np.float32)
    engine = build_engine(MODEL_PROTOTXT, CAFFE_MODEL, verbose=False)
    # import pdb
    # pdb.set_trace()
    # with
    with open('models/R50_SOFTMAX_VERT.engine', 'wb') as f:
        f.write(engine.serialize())
    #img = cv2.imread('33347.png')
    # img_resized =  _preprocess_img(img, (384, 288))
    img_resized = np.load('data.npy')
    # img_resized = np.zeros_like(img_resized).astype(np.float32)
    # with open('models/R50_SOFTMAX.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    #    engine = runtime.deserialize_cuda_engine(f.read())
    # #
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = alloc_buf(engine)
    inputs[0].host = np.ascontiguousarray(img_resized)

    for _ in range(1):
        t1 = time.time()

        trt_outputs = do_inference(
            context=context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream)
        print("cost time: ", time.time()-t1)
    import pdb
    pdb.set_trace()
    #for _ in range(1):
    #   t1 = time.time()
    #   in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
    #   res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
    #   print(res)
    #   print("cost time: ", time.time()-t1)


# tensorrt docker image: docker pull nvcr.io/nvidia/tensorrt:19.09-py3 (See: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags)
# NOTE: cuda driver >= 418
