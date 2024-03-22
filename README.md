# deeplab-ncnn-demo

1. onnx2ncnn:
  bin/onnx2ncnn model_param/deeplab_torch_quantized.onnx model_param/deeplab_torch_quantized.param model_param/deeplab_torch_quantized.bin

2. cmake files
   cd cmake
   cmake CMakeLists.txt
   make
   
3. inference
   bin/deeplab image/bus.jpg
   
4.inference file
  deeplab.cpp

5.quantize tools
  ppq
  ncnn
