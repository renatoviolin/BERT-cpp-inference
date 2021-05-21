#include <vector>

#include "torch/script.h"

torch::Tensor predict(torch::Tensor input_ids,
                      torch::jit::script::Module module);

// std::vector<float> predict(torch::Tensor input_ids,
//                            torch::Tensor segment_ids,
//                            torch::jit::script::Module module);