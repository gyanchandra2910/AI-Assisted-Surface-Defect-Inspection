#pragma once
#include <filesystem>
#include <opencv2/geometry.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
namespace ai {
namespace fs = std::filesystem;
struct Config {
    int version = 1, width = 224, height = 608;
    bool residualFeatures = true;
    double threshold = 0.5, orientation = 1;
    std::string kind = "rf";
    int localThreshold = 18, minArea = 12, margin = 8;
    int denoiseKernel = 3, claheGridX = 8, claheGridY = 16, morphologyKernel = 17, patchSize = 24;
    double denoiseSigma = 0.7, claheClip = 2.0, backgroundSigma = 5.0, gradientSupport = 4.0;
    void write(cv::FileStorage &f) const;
    void read(const cv::FileNode &n);
};
struct Stages {
    cv::Mat gray, denoised, enhanced, gradient, residual, processed;
};
struct Region {
    std::vector<cv::Point> contour;
    cv::Rect box;
    int pixelArea = 0;
};
struct Result {
    float score = 0;
    bool defective = false;
    cv::Mat processed, mask;
    std::vector<Region> regions;
    double featureMs = 0, predictMs = 0, localizeMs = 0;
};
Stages preprocess(const cv::Mat &image, const Config &c);
cv::Mat features(const cv::Mat &image, const Config &c);
cv::Mat localize(const cv::Mat &image, const Config &c, std::vector<Region> *regions = nullptr);
class Model {
  public:
    Config config;
    cv::Mat mean, scale;
    cv::Ptr<cv::ml::StatModel> classifier;
    cv::Mat normalize(const cv::Mat &rows) const;
    void fitScale(const cv::Mat &train);
    void train(const cv::Mat &x, const cv::Mat &y, std::string kind, double parameter, int seed);
    std::vector<float> scores(const cv::Mat &x) const;
    Result analyze(const cv::Mat &image) const;
    void save(const fs::path &dir) const;
    void load(const fs::path &dir);
};
double milliseconds(std::chrono::steady_clock::time_point start);
struct Sample {
    fs::path image, mask;
    std::string split, id, hash;
    int label = 0, width = 0, height = 0;
};
std::vector<Sample> manifest(const fs::path &data, const fs::path &output);
void trainEvaluate(const fs::path &data, const fs::path &output);
void evaluateSaved(const fs::path &data, const fs::path &model, const fs::path &output);
} // namespace ai
