#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>


static const char* keys = {
    "{help h usage ? | | print this message   }"
    "{img  | ../monodepth2/assets/test_image.jpg | Path to input image }"
    "{out  | ../demo.jpg | Path to output image }"
    "{pth  | | Absolute path to the folder containing *.pth }"
};

static const std::string message =
    "\nThis demo estimates the depth map given an input image."
    "\nFor further information, please refer to https://github.com/nianticlabs/monodepth2."
    "\n";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about(message);
    std::string img_path, pth_path, save_path;

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("img"))
    {
        img_path = parser.get<std::string>("img");
    }
    if (parser.has("out"))
    {
        save_path = parser.get<std::string>("out");
    }
    if (parser.has("pth"))
    {
        pth_path = parser.get<std::string>("pth");
    }

    if (img_path.empty() || pth_path.empty() || save_path.empty())
    {
        std::cerr << "Missing arguments!\n";
        parser.printMessage();
        return -1;
    }

    // Define input image size
    int w = 640;
    int h = 192;

    // Load models
    torch::DeviceType device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::jit::script::Module resnet_encoder, depth_decoder;

    resnet_encoder = torch::jit::load(pth_path + "/encoder.jit.pth");
    resnet_encoder.eval();
    resnet_encoder.to(device);

    depth_decoder = torch::jit::load(pth_path + "/depth.jit.pth");
    depth_decoder.eval();
    depth_decoder.to(device);
    
    // Read ipout image
    cv::Mat img = cv::imread(img_path);
    cv::Mat norm;
    cv::resize(img, norm, cv::Size(w, h));
    norm.convertTo(norm, CV_32FC3, 1. / 255.);

    torch::Tensor inputs = torch::from_blob(norm.data, {1, norm.rows, norm.cols, 3}, torch::kF32);
    inputs = inputs.permute({0, 3, 1, 2});
    inputs = inputs.to(device);

    // Inference
    auto outputs = depth_decoder.forward({resnet_encoder.forward({inputs})});
    auto disp_tensor = outputs.toGenericDict().at("disp0").toTensor().to(at::kCPU).permute({0, 3, 2, 1});

    double min_val, max_val;
    cv::Mat disp = cv::Mat(h, w, CV_32FC1, disp_tensor.data_ptr());
    cv::resize(disp, disp, cv::Size(img.cols, img.rows));
    cv::minMaxLoc(disp, &min_val, &max_val);
    disp = 255 * (disp - min_val) / (max_val - min_val);
    disp.convertTo(disp, CV_8U);

    if ((CV_VERSION_MAJOR == 3 && CV_VERSION_MINOR <= 4 && CV_VERSION_REVISION < 6) ||
        (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 1))
    {
        cv::applyColorMap(disp, disp, 2);
    }
    else
    {
        cv::applyColorMap(disp, disp, 13);
    }

    img.push_back(disp);
    cv::imwrite(save_path, img);
}