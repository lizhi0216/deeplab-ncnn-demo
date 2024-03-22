#include "net.h"
#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif
#include <stdio.h>
#include <vector>

//保留比例的放大，剩下边加灰条
std::tuple<cv::Mat, int, int> resize_image(const cv::Mat& inputImage, cv::Size size) {
    int iw = inputImage.cols;
    int ih = inputImage.rows;
    int w = size.width;
    int h = size.height;

    double scale = std::min(static_cast<double>(w) / iw, static_cast<double>(h) / ih);
    int nw = static_cast<int>(iw * scale);
    int nh = static_cast<int>(ih * scale);

    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, cv::Size(nw, nh), 0, 0, cv::INTER_CUBIC);

    cv::Mat newImage = cv::Mat::ones(size, inputImage.type())*cv::Scalar(128, 128, 128);
    cv::Rect roi((w - nw) / 2, (h - nh) / 2, nw, nh);
    resizedImage.copyTo(newImage(roi));

    return std::make_tuple(newImage, nw, nh);
}


// static int detect_resnet18(const cv::Mat& image, std::vector<float>& cls_scores)
static cv::Mat inference(const cv::Mat& image)
{

    int img_w = image.cols;
    int img_h = image.rows;

    //实例化一个类net，名字是deeplab
    ncnn::Net deeplab;
    
    deeplab.opt.use_vulkan_compute = true;

    //分别加载模型的参数和数据，读取参数后可以自动生成模型
    if (deeplab.load_param("model_param/deeplab_torch_quantized.param"))
        exit(-1);
    if (deeplab.load_model("model_param/deeplab_torch_quantized.bin"))
        exit(-1);
    
	//opencv读取图片是BGR格式，我们需要转换为RGB格式
    //ncnn::Mat 存储处理后图像数据的类
    //from_pixels_resize函数参数，输入数据地址，转换格式，输入图片长宽，输出图片长宽
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, 512, 512);
    //图像归一标准化，
    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};//对rgb三个通道进行归一标准化
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    input.substract_mean_normalize(mean_vals, norm_vals);
    //Extractor类用于从网络中提取特定层的输出
    ncnn::Extractor ex = deeplab.create_extractor();
	
    //把图像数据放入input.1这个blob里, input.1是网络最初的输入名称
    ex.input("images", input);

    ncnn::Mat output;
    //提取出推理结果，推理结果存放在191这个blob里，191应该是原网络中最终输出的名字
    ex.extract("output", output);
    ncnn::Mat out_image = output.clone();

    // //pytorch预测出的out是1，3，512，512，取out【0】做预测值映射出颜色
    // fprintf(stderr, "%d\n", output.c);//3
    // fprintf(stderr, "%d\n", output.w);//512
    // fprintf(stderr, "%d\n", output.h);//512
    // fprintf(stderr, "%d\n", output.dims);//3

    //softmax
    // 创建一个softmax层
    ncnn::Layer* softmax = ncnn::create_layer("Softmax");
    // 设置softmax的参数
    ncnn::ParamDict pd;
    pd.set(0, 1); // axis
    // 前向运行softmax层
    softmax->load_param(pd);
    softmax->forward_inplace(out_image, ncnn::Option());
    // 此时，output已经被softmax转换
    //使用后删除softmax，防止内存泄漏
    delete softmax;

    // 进行图片的resize
    ncnn::Mat output_resized;
    ncnn::resize_bilinear(output, output_resized, img_w, img_h);

    // 取出每一个像素点的种类
    ncnn::Mat output_argmax(output_resized.w, output_resized.h, output_resized.elemsize);
    for (int i = 0; i < output_resized.h; i++)
    {
        for (int j = 0; j < output_resized.w; j++)
        {
            float max_val = output_resized.channel(0).row(i)[j];
            int max_idx = 0;
            for (int q = 1; q < output_resized.c; q++)
            {
                float val = output_resized.channel(q).row(i)[j];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = q;
                }
            }
            output_argmax.row(i)[j] = max_idx;
        }
    }

    // 为每一种类像素赋颜色
    std::vector<cv::Vec3b> colors = { cv::Vec3b{0, 0, 0}, cv::Vec3b{255, 0, 0}, cv::Vec3b{0, 255, 0}, cv::Vec3b{128, 128, 0}, cv::Vec3b{0, 0, 128}, cv::Vec3b{128, 0, 128}, cv::Vec3b{0, 128, 128},
                            cv::Vec3b{128, 128, 128}, cv::Vec3b{64, 0, 0}, cv::Vec3b{192, 0, 0}, cv::Vec3b{64, 128, 0}, cv::Vec3b{192, 128, 0}, cv::Vec3b{64, 0, 128}, cv::Vec3b{192, 0, 128}, 
                            cv::Vec3b{64, 128, 128}, cv::Vec3b{192, 128, 128}, cv::Vec3b{0, 64, 0}, cv::Vec3b{128, 64, 0}, cv::Vec3b{0, 192, 0}, cv::Vec3b{128, 192, 0}, cv::Vec3b{0, 64, 128}, 
                            cv::Vec3b{128, 64, 12}};


    cv::Mat seg_img(img_h, img_w, CV_8UC3);
    for (int i = 0; i < seg_img.rows; i++)
    {
        for (int j = 0; j < seg_img.cols; j++)
        {
            seg_img.at<cv::Vec3b>(i, j) = colors[output_argmax.row(i)[j]];
        }
    }
    cv::Mat bgr_image = seg_img; // 这是你的BGR图像
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

    // // 将新图与原图及进行混合
    // cv::Mat old_img = 最初的输入 ; // 这是你的原图
    // cv::Mat image = old_img * 0 + seg_img * 1; // alpha=1，只保留掩膜图

    return rgb_image;
}


int main(int argc, char** argv)
{
    
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    cv::Mat out = inference(m);
    cv::imwrite("mask.jpg", out);

    return 0;
}














// // 假设pr是一个三维的ncnn::Mat对象
// ncnn::Mat pr = ...;

// // 对每一行进行softmax转换
// for (int q = 0; q < pr.c; q++)
// {
//     ncnn::Mat m = pr.channel(q);
//     float sum = 0.f;
//     for (int i = 0; i < m.w * m.h; i++)
//     {
//         m[i] = exp(m[i]);
//         sum += m[i];
//     }
//     for (int i = 0; i < m.w * m.h; i++)
//     {
//         m[i] /= sum;
//     }
// }

// // 将灰条部分截取掉
// int nh = ...; // 需要截取的高度
// int nw = ...; // 需要截取的宽度
// pr = pr(ncnn::Rect((pr.w - nw) / 2, (pr.h - nh) / 2, nw, nh));

// // 进行图片的resize
// int orininal_w = ...; // 原始宽度
// int orininal_h = ...; // 原始高度
// ncnn::Mat pr_resized;
// ncnn::resize_bilinear(pr, pr_resized, orininal_w, orininal_h);

// // 取出每一个像素点的种类
// ncnn::Mat pr_argmax(pr_resized.w, pr_resized.h, pr_resized.elemsize);
// for (int i = 0; i < pr_resized.h; i++)
// {
//     for (int j = 0; j < pr_resized.w; j++)
//     {
//         float max_val = pr_resized.channel(0).row(i)[j];
//         int max_idx = 0;
//         for (int q = 1; q < pr_resized.c; q++)
//         {
//             float val = pr_resized.channel(q).row(i)[j];
//             if (val > max_val)
//             {
//                 max_val = val;
//                 max_idx = q;
//             }
//         }
//         pr_argmax.row(i)[j] = max_idx;
//     }
// }

// // 为每一种类像素赋颜色
// std::vector<cv::Vec3b> colors = ...; // 这是你的颜色表
// cv::Mat seg_img(orininal_h, orininal_w, CV_8UC3);
// for (int i = 0; i < seg_img.rows; i++)
// {
//     for (int j = 0; j < seg_img.cols; j++)
//     {
//         seg_img.at<cv::Vec3b>(i, j) = colors[pr_argmax.row(i)[j]];
//     }
// }

// // 将新图与原图及进行混合
// cv::Mat old_img = ...; // 这是你的原图
// cv::Mat image = old_img * 0 + seg_img * 1; // alpha=1，只保留掩膜图

