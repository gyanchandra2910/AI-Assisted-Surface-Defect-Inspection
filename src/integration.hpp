#pragma once
#include "geometry.hpp"
#include "pipeline.hpp"
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
namespace app {
struct Packet {
    uint64_t token = 0;
    std::string source, error;
    int frame = -1, totalFrames = 0;
    double fps = 0, decodeMs = 0;
    cv::Mat image;
    ai::Result result;
};
class Processor {
    struct Job {
        uint64_t token;
        int type, frame;
        std::string path;
    };
    ai::Model model;
    std::mutex mutex;
    std::condition_variable condition;
    bool stopping = false;
    std::optional<Job> pending;
    std::optional<Packet> completed;
    uint64_t requested = 0;
    std::thread worker;
    void run();

  public:
    explicit Processor(const ai::fs::path &modelDir);
    ~Processor();
    uint64_t request(int type, const std::string &path = "",
                     int frame = 0); // 0 image, 1 open video, 2 next, 3 seek
    std::optional<Packet> take();
    const ai::Config &config() const {
        return model.config;
    }
};
void makeDemo(const ai::fs::path &data, const ai::fs::path &output);
void verifyRuntime(const ai::fs::path &data, const ai::fs::path &model, const ai::fs::path &video,
                   const ai::fs::path &output);
int viewer(const ai::fs::path &data, const ai::fs::path &model, const std::string &image,
           bool smoke, const ai::fs::path &output);
} // namespace app
