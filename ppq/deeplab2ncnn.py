import os
from typing import Iterable
import torch
import torchvision
from ppq import *
from ppq.api import *
import numpy as np
import torch
import onnx
import torch
from pathlib import Path

#ncnn中有些算子不支持，要用onnxsim简化

INPUT_SHAPE = [1, 3, 512, 512]
model_path = '/home/disk1/fxq/lianghua/ppq-master/deeplab_handle/logs/deeplab_torch_quantized.onnx'
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model = model.to('cuda:0')
model.eval()
input_layer_names = ["images"]
output_layer_names = ["output"]
im = torch.zeros(*INPUT_SHAPE).to('cuda:0')
torch.onnx.export(model,
                im,#模型的输入参数，通常是一个tensor，但里面的值没影响只取形状
                f=model_path,
                verbose=False,#控制打印信息的详细程度
                opset_version=12,
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,#训练时不折叠常量，推理时折叠提高效率
                input_names=input_layer_names,
                output_names=output_layer_names,
                dynamic_axes=None)

# Checks这两行代码通常用于加载一个ONNX模型，并在使用模型之前验证其有效性
model_onnx = onnx.load(model_path)
onnx.checker.check_model(model_onnx)

# Simplify onnx

import onnxsim
print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
model_onnx, check = onnxsim.simplify(
    model_onnx,
    dynamic_input_shape=False,#输入图形状是否固定
    input_shapes=None)#自动推断输入尺寸
assert check, 'assert check failed'
onnx.save(model_onnx, model_path)


print('Onnx model save as {}'.format(model_path))


# model_path = '/home/disk1/fxq/lianghua/ppq-master/deeplab_xception_nosim.onnx' # onnx simplified model
# model_path = '/home/disk1/fxq/lianghua/ppq-master/deeplab_handle/logs/deeplab_xception_nosim.onnx' # onnx unsimplified model
data_path  = '/data/ImageNet/calibration' # calibration data folder


# initialize dataloader, suppose preprocessed calibration data is in binary format
WORKING_DIRECTORY = '/home/disk1/fxq/lianghua/ppq-master/T_handle_dataset/JPEGImages'
CALIBRATION_BATCHSIZE = 1

INPUT_LAYOUT = 'hwc' # input data layout, chw or hwc, channel  height  width
# npy_array = [np.fromfile(os.path.join(data_path, file_name), dtype=np.float32).reshape(*INPUT_SHAPE) for file_name in os.listdir(data_path)]
# dataloader = [torch.from_numpy(np.load(npy_tensor)) for npy_tensor in npy_array]


EXECUTING_DEVICE = 'cuda:0'
# EXECUTING_INPUT_SHAPE = [32, 3, 512, 512]

# confirm platform and setting
target_platform = TargetPlatform.NCNN_INT8
setting = QuantizationSettingFactory.default_setting()
# setting.dispatching_table.append('/backbone/block9/sepconv1/depthwise/Conv', platform=TargetPlatform.FP32)



# setting.quantize_parameter

def load_calibration_dataset()->Iterable:
    return [torch.rand(INPUT_SHAPE) for _ in range(32)]
dataloader = load_calibration_dataset()
# dataloader = load_calibration_dataset(
#     directory    = WORKING_DIRECTORY,
#     input_shape  = INPUT_SHAPE,
#     batchsize    = CALIBRATION_BATCHSIZE,
#     input_format = INPUT_LAYOUT)

def collate_fn(batch: torch.Tensor)->torch.Tensor:
    return batch.to(EXECUTING_DEVICE)

calib_steps = 8#max(min(512, len(dataloader)), 8)

ppq_graph_ir = quantize_onnx_model(onnx_import_file=model_path, calib_dataloader=dataloader, calib_steps=calib_steps, input_shape=INPUT_SHAPE,
                                   platform=target_platform, setting=setting, collate_fn=collate_fn)

print('end')
reports = layerwise_error_analyse(graph=ppq_graph_ir, running_device=EXECUTING_DEVICE, collate_fn=collate_fn, dataloader=dataloader)

reports = graphwise_error_analyse(graph=ppq_graph_ir, running_device=EXECUTING_DEVICE, collate_fn=collate_fn, dataloader=dataloader)


export_ppq_graph(graph=ppq_graph_ir, platform=target_platform, graph_save_to='deeplab_torch_quantized.onnx')




