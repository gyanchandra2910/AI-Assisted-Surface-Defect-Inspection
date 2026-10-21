#include "integration.hpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        cv::setNumThreads(1);
        std::string mode = argc > 1 ? argv[1] : "view", image;
        ai::fs::path data = "data/KolektorSDD2", out = "outputs", model = "outputs/model";
        bool smoke = false;
        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            auto value = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("Missing value for " + a);
                return std::string(argv[i]);
            };
            if (a == "--data")
                data = value();
            else if (a == "--output")
                out = value();
            else if (a == "--model")
                model = value();
            else if (a == "--image")
                image = value();
            else if (a == "--smoke")
                smoke = true;
            else
                throw std::runtime_error("Unknown argument: " + a);
        }
        if (mode == "prepare")
            ai::manifest(data, out);
        else if (mode == "train")
            ai::trainEvaluate(data, out);
        else if (mode == "evaluate")
            ai::evaluateSaved(data, model, out);
        else if (mode == "demo-video")
            app::makeDemo(data, out / "demo_sequence.avi");
        else if (mode == "verify")
            app::verifyRuntime(data, model, out / "demo_sequence.avi", out);
        else if (mode == "predict") {
            if (image.empty())
                throw std::runtime_error("predict requires --image");
            ai::Model m;
            m.load(model);
            auto r = m.analyze(cv::imread(image));
            std::cout << (r.defective ? "defective" : "normal") << ", score=" << r.score
                      << ", threshold=" << m.config.threshold
                      << ", candidate regions=" << r.regions.size() << '\n';
        } else if (mode == "view")
            return app::viewer(data, model, image, smoke, out);
        else if (mode == "help" || mode == "--help")
            std::cout << "inspect prepare|train|evaluate|view|predict|demo-video|verify [--data "
                         "PATH] [--output PATH] [--model PATH] [--image PATH] [--smoke]\n";
        else
            throw std::runtime_error("Unknown command: " + mode);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
