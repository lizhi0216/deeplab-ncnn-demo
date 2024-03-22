# deeplab-ncnn-demo

1. onnx2ncnn:  
  bin/onnx2ncnn model_param/deeplab_torch_quantized.onnx model_param/deeplab_torch_quantized.param model_param/deeplab_torch_quantized.bin
  
  
2. cmake files：   
   cd cmake
   cmake CMakeLists.txt  
   make  
     
3. inference：  
   bin/deeplab image/bus.jpg
   
5. inference file：  
   src/deeplab.cpp
   
7. quantize tools：  
   ppq/deeplab2ncnn.py (onnx quantize)
   ppq/deeplabtest.py (torch model quantize)  
   ncnnint8 (platform)
   
9. model_param  
   pretrained_weight.onnx  
   pretrained_weight.param  
   pretrained_weight.bin  
     
