#include "infer.h"

torch::Tensor predict(torch::Tensor input_ids,
                      torch::jit::script::Module module) {
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inp_tokens{input_ids};

    auto output = module.forward(inp_tokens);
    auto preds = output.toTuple()->elements()[0];  // last_hidden_state

    // Convert tensor to vector
    return preds.toTensor()[0];  // (seq_len, 30522)
}

// #include "infer.h"

// std::vector<float> predict(torch::Tensor input_ids,
//                            torch::Tensor segment_ids,
//                            torch::jit::script::Module module) {
//     // auto tokens = torch::tensor({{101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 102}});
//     // auto segments = torch::tensor({{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1}});

//     // std::vector<torch::jit::IValue> inp_tokens{input_ids};
//     // std::vector<torch::jit::IValue> inp_segments{segment_ids};

//     torch::NoGradGuard no_grad;
//     c10::detail::ListImpl::list_type input_list{input_ids, segment_ids};

//     auto output = module.forward(input_list);
//     auto last_hidden = output.toTuple()->elements()[0];    // last_hidden_state
//     auto pooler_output = output.toTuple()->elements()[1];  // pooler_output
//     std::cout << pooler_output << std::endl;

//     // Convert tensor to vector
//     auto p = pooler_output.toTensor().data_ptr<float>();
//     std::vector<float> result{p, p + pooler_output.toTensor().size(1)};
//     std::cout << result << std::endl;

//     return result;
// }
