from typing import Iterable
import torch
import torchvision
from ppq import *
from ppq.api import *
from deeplab_handle.nets.deeplabv3_plus import DeepLab

INPUT_SHAPE = [8, 3, 512, 512]
DEVICE = 'cuda:0'
PLATFORM = TargetPlatform.NCNN_INT8

WORKING_DIRECTORY = '/home/disk1/fxq/lianghua/ppq-master/T_handle_dataset/JPEGImages'
CALIBRATION_BATCHSIZE = 1
# INPUT_SHAPE = [1, 3, 512, 512]
INPUT_LAYOUT = 'hwc' # input data layout, chw or hwc, channel  height  width


def load_calibration_dataset()->Iterable:
    return [torch.rand(INPUT_SHAPE) for _ in range(32)]
dataloader = load_calibration_dataset()

# dataloader = load_calibration_dataset(
#     directory    = WORKING_DIRECTORY,
#     input_shape  = INPUT_SHAPE,
#     batchsize    = CALIBRATION_BATCHSIZE,
#     input_format = INPUT_LAYOUT)


def collate_fn(batch: torch.Tensor)->torch.Tensor:

    return batch.to(DEVICE)


num_classes = 3
input_shape = [512, 512]
backbone = "xception"  # backbone：mobilenet、xception
downsample_factor = 16  # downsample factor, same with training
pretrained = True
# model_path = '/home/disk1/fxq/lianghua/ppq-master/deeplab_handle/model_data/deeplab_xception.pth'  # *.pth model path
# model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

# model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
#                     pretrained=pretrained)

# model.load_state_dict(torch.load(model_path, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

setting = QuantizationSettingFactory.ncnn_setting()
# setting.dispatching_table.append('/backbone/block9/sepconv1/depthwise/Conv', platform=TargetPlatform.FP32)


calib_steps = max(min(512, len(dataloader)), 8)
ir = quantize_torch_model(
    model=model, calib_dataloader=dataloader, platform=PLATFORM, setting=setting, calib_steps=calib_steps, input_shape=INPUT_SHAPE, collate_fn=collate_fn
)


reports = layerwise_error_analyse(graph=ir, running_device=DEVICE, collate_fn=collate_fn, dataloader=dataloader)

reports = graphwise_error_analyse(graph=ir, running_device=DEVICE, collate_fn=collate_fn, dataloader=dataloader)


export_ppq_graph(graph=ir, platform=PLATFORM, graph_save_to='/home/disk1/fxq/lianghua/ppq-master/model_param/deeplab_torch_quantized.onnx')

# import onnx
# import onnxsim
# print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
# model_onnx = onnx.load('/home/disk1/fxq/lianghua/ppq-master/model_param/deeplab_torch_quantized.onnx')
# onnx.checker.check_model(model_onnx)
# model_onnx, check = onnxsim.simplify(
#     model_onnx,
#     dynamic_input_shape=False,#输入图形状是否固定
#     input_shapes=None)#自动推断输入尺寸
# assert check, 'assert check failed'
# onnx.save(model_onnx, '/home/disk1/fxq/lianghua/ppq-master/model_param/deeplab_torch_quantized.onnx')