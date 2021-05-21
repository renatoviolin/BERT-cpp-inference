// #include "libs/infer.h"
// #include "libs/crow_all.h"
// #include "libs/base64.h"

// int PORT = 8000;

// int main(int argc, char **argv)
// {

//     if (argc != 4)
//     {
//         std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> \n";
//         return -1;
//     }

//     std::string model_path = argv[1];
//     std::string labels_path = argv[2];
//     std::string usegpu_str = argv[3];
//     bool usegpu;

//     if (usegpu_str == "true")
//         usegpu = true;
//     else
//         usegpu = false;

//     // Set image height and width
//     int image_height = 224;
//     int image_width = 224;

//     // Read labels
//     std::vector<std::string> labels;
//     std::string label;
//     std::ifstream labelsfile(labels_path);
//     {
//         while (getline(labelsfile, label))
//             labels.push_back(label);
//         labelsfile.close();
//     }

//     // Define mean and std
//     std::vector<double> mean = {0.485, 0.456, 0.406};
//     std::vector<double> std = {0.229, 0.224, 0.225};

//     // Load Model
//     torch::jit::script::Module model = read_model(model_path, usegpu);

//     // App
//     crow::SimpleApp app;
//     CROW_ROUTE(app, "/predict").methods("POST"_method, "GET"_method)([&image_height, &image_width, &mean, &std, &labels, &model, &usegpu](const crow::request &req) {
//         crow::json::wvalue result;
//         result["Prediction"] = "";
//         result["Confidence"] = "";
//         result["Status"] = "Failed";
//         std::ostringstream os;

//         try
//         {
//             auto args = crow::json::load(req.body);

//             // Get Image
//             std::string base64_image = args["image"].s();
//             int request_id = (int)args["request_id"];
//             std::cout << "Request id: " << request_id << std::endl;

//             std::string decoded_image = base64_decode(base64_image);
//             std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
//             cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

//             // Predict
//             std::string pred, prob;
//             tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);

//             result["Prediction"] = pred;
//             result["Confidence"] = prob;
//             result["Status"] = "Success";

//             os << crow::json::dump(result);
//             return crow::response{os.str()};
//         }
//         catch (std::exception &e)
//         {
//             os << crow::json::dump(result);
//             return crow::response(os.str());
//         }
//     });

//     // app.port(PORT).run();
//     app.port(PORT).multithreaded().run();
//     return 0;
// }

#include <vector>

#include "torch/script.h"

int main(int argc, char** argv) {
    auto tokens = torch::tensor({{101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 102}});
    auto segments = torch::tensor({{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1}});

    std::vector<torch::jit::IValue> inp_tokens{tokens};
    std::vector<torch::jit::IValue> inp_segments{segments};

    auto module = torch::jit::load(argv[1]);
    c10::detail::ListImpl::list_type input_list{tokens, segments};

    module.eval();
    torch::NoGradGuard no_grad;

    auto output = module.forward(input_list);
    auto last_hidden = output.toTuple()->elements()[0];    // last_hidden_state
    auto pooler_output = output.toTuple()->elements()[1];  // pooler_output
    std::cout << pooler_output << std::endl;

    // Convert tensor to vector
    auto p = pooler_output.toTensor().data_ptr<float>();
    std::vector<float> result{p, p + pooler_output.toTensor().size(1)};
    std::cout << result << std::endl;

    // Mat image = imread(argv[2]);
    // image = preprocess(image, 224, 224, _mean, _std);

    // torch::Tensor image_as_tensor;
    // image_as_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat32).clone();
    // image_as_tensor = image_as_tensor.permute({0, 3, 1, 2});

    // // std::cout << image_as_tensor.options() << std::endl;
    // // int d = image_as_tensor.dim();
    // // for (int i = 0; i < d; i++)
    // // {
    // //     std::cout << image_as_tensor.size(i) << " ";
    // // }

    // std::cout << image_as_tensor << std::endl;

    // auto module = torch::jit::load(argv[1]);
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(image_as_tensor);
    // torch::NoGradGuard no_grad;
    // torch::Tensor output = module.forward(inputs).toTensor();

    // std::cout << output << std::endl;
    // std::cout << torch::softmax(output, 1) << std::endl;
    // std::cout << output.argmax(1) << std::endl;

    // torch::Tensor logits, index;
    // std::tie(logits, index) = torch::softmax(output, 1).topk(5);
    // // std::cout << logits.data() << std::endl;
    // // std::cout << index << std::endl;

    // std::vector<float> scores;
    // std::cout << logits.size(1) << std::endl;

    // for (int i = 0; i < logits.size(1); i++)
    // {
    //     scores.push_back(*logits[0][i].data_ptr<float>());
    // }

    // for (int i = 0; i < scores.size(); i++)
    // {
    //     printf("%.5f ", scores[i]);
    // }

    return 0;
}
