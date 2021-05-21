#include <boost/algorithm/string.hpp>
#include <iostream>

#include "libs/crow_all.h"
#include "libs/infer.h"
#include "torch/script.h"

int PORT = 8000;

int main(int argc, char **argv) {
    crow::SimpleApp app;

    // torch::jit::script::Module module = torch::jit::load("../model/large_lm.pt");
    torch::jit::script::Module module = torch::jit::load("../model/large_lm.pt");
    module.eval();

    CROW_ROUTE(app, "/predict").methods("POST"_method)([&module](const crow::request &req) {
        crow::json::wvalue result;
        result["Prediction"] = "";
        result["Confidence"] = "";
        std::ostringstream os;

        try {
            auto args = crow::json::load(req.body);
            std::string tokens_text = args["tokens"].s();
            int idx_to_predict = (int)args["idx_to_predict"];
            int request_id = (int)args["request_id"];

            std::vector<std::string> tokens;
            std::vector<int32_t> token_id;
            boost::split(tokens, tokens_text, boost::is_any_of(","));
            for (auto &x : tokens) {
                token_id.push_back(std::stoi(x));
            }

            // convert to tensor
            auto th_tokens = torch::from_blob(token_id.data(), {1, long(token_id.size())}, torch::kInt32);
            auto r = predict(th_tokens, module);
            auto probs = r[idx_to_predict].softmax(0);

            torch::Tensor logits, idx;
            std::tie(logits, idx) = probs.topk(5);
            std::vector<int64_t> p{idx.data_ptr<int64_t>(), idx.data_ptr<int64_t>() + idx.size(0)};
            std::vector<float_t> l{logits.data_ptr<float_t>(), logits.data_ptr<float_t>() + logits.size(0)};

            result["Prediction"] = p;
            result["Confidence"] = l;

            os << crow::json::dump(result);
            return crow::response{os.str()};

        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
            os << crow::json::dump(result);
            return crow::response(os.str());
        }
    });

    // app.port(PORT).run();
    app.port(PORT).multithreaded().run();
    return 0;
}
