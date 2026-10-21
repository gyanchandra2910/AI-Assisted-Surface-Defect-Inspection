#include "pipeline.hpp"
#include <chrono>
#include <fstream>
#include <numeric>
#include <stdexcept>
namespace ai {
double milliseconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
void Config::write(cv::FileStorage &f) const {
    f << "configuration" << "{" << "version" << version << "width" << width << "height" << height
      << "residual_features" << int(residualFeatures) << "kind" << kind << "threshold" << threshold
      << "orientation" << orientation << "local_threshold" << localThreshold << "min_area"
      << minArea << "margin" << margin << "denoise_kernel" << denoiseKernel << "denoise_sigma"
      << denoiseSigma << "clahe_clip" << claheClip << "clahe_grid_x" << claheGridX << "clahe_grid_y"
      << claheGridY << "morphology_kernel" << morphologyKernel << "background_sigma"
      << backgroundSigma << "gradient_support" << gradientSupport << "patch_size" << patchSize
      << "}";
}
void Config::read(const cv::FileNode &n) {
    n["version"] >> version;
    n["width"] >> width;
    n["height"] >> height;
    int v;
    n["residual_features"] >> v;
    residualFeatures = v;
    n["kind"] >> kind;
    n["threshold"] >> threshold;
    n["orientation"] >> orientation;
    n["local_threshold"] >> localThreshold;
    n["min_area"] >> minArea;
    n["margin"] >> margin;
    n["denoise_kernel"] >> denoiseKernel;
    n["denoise_sigma"] >> denoiseSigma;
    n["clahe_clip"] >> claheClip;
    n["clahe_grid_x"] >> claheGridX;
    n["clahe_grid_y"] >> claheGridY;
    n["morphology_kernel"] >> morphologyKernel;
    n["background_sigma"] >> backgroundSigma;
    n["gradient_support"] >> gradientSupport;
    n["patch_size"] >> patchSize;
    if (version != 1 || width < 96 || height < 144 || patchSize != 24 || denoiseKernel < 1 ||
        !(denoiseKernel % 2) || morphologyKernel < 1 || !(morphologyKernel % 2) || claheGridX < 1 ||
        claheGridY < 1 || margin < 0 || margin >= width / 2 || backgroundSigma <= 0)
        throw std::runtime_error("Unsupported model configuration");
}
Stages preprocess(const cv::Mat &image, const Config &c) {
    if (image.empty() || image.depth() != CV_8U)
        throw std::runtime_error("Image must be a decoded 8-bit image");
    Stages s;
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
    else
        throw std::runtime_error("Unsupported image channels");
    cv::resize(gray, s.gray, {c.width, c.height}, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(s.gray, s.denoised, {c.denoiseKernel, c.denoiseKernel}, c.denoiseSigma);
    cv::createCLAHE(c.claheClip, {c.claheGridX, c.claheGridY})->apply(s.denoised, s.enhanced);
    cv::Mat gx, gy;
    cv::Sobel(s.enhanced, gx, CV_32F, 1, 0, 3);
    cv::Sobel(s.enhanced, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, s.gradient);
    s.gradient /= 4.0;
    cv::Mat background;
    cv::GaussianBlur(s.denoised, background, {0, 0}, c.backgroundSigma);
    cv::absdiff(s.denoised, background, s.residual);
    s.processed = s.enhanced;
    return s;
}
static void stats(const cv::Mat &input, std::vector<float> &out) {
    cv::Mat f;
    input.convertTo(f, CV_32F);
    cv::Scalar mean, sd;
    cv::meanStdDev(f, mean, sd);
    out.push_back(float(mean[0] / 255));
    out.push_back(float(sd[0] / 255));
    std::vector<float> a;
    a.assign(f.begin<float>(), f.end<float>());
    std::sort(a.begin(), a.end());
    for (double q : {0.5, 0.9, 0.99, 0.999})
        out.push_back(a[std::min(a.size() - 1, size_t(q * (a.size() - 1)))] / 255);
    for (float t : {8.f, 16.f, 32.f})
        out.push_back(float(cv::countNonZero(f > t)) / f.total());
}
cv::Mat features(const cv::Mat &image, const Config &c) {
    auto s = preprocess(image, c);
    std::vector<float> out;
    stats(s.denoised, out);
    stats(s.enhanced, out);
    stats(s.gradient, out);
    // Local residual statistics capture small spots and scratches without target masks or ID
    // features.
    std::vector<float> texture;
    stats(s.residual, texture);
    cv::Mat dark, bright;
    auto kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {c.morphologyKernel, c.morphologyKernel});
    cv::morphologyEx(s.denoised, dark, cv::MORPH_BLACKHAT, kernel);
    cv::morphologyEx(s.denoised, bright, cv::MORPH_TOPHAT, kernel);
    stats(dark, texture);
    stats(bright, texture);
    for (const auto &m : {s.residual, s.gradient, dark, bright}) {
        std::vector<float> means, peaks;
        for (int y = 8; y + c.patchSize <= m.rows - 8; y += c.patchSize)
            for (int x = 8; x + c.patchSize <= m.cols - 8; x += c.patchSize) {
                cv::Mat p = m(cv::Rect(x, y, c.patchSize, c.patchSize));
                means.push_back(float(cv::mean(p)[0]));
                double mx;
                cv::minMaxLoc(p, nullptr, &mx);
                peaks.push_back(float(mx));
            }
        std::sort(means.rbegin(), means.rend());
        std::sort(peaks.rbegin(), peaks.rend());
        for (int k : {0, 1, 4, 9}) {
            texture.push_back(means[k] / 255);
            texture.push_back(peaks[k] / 255);
        }
    }
    // Spatial intensity statistics use fixed, label-independent cells across the complete image.
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 3; x++) {
            cv::Rect r(x * c.width / 3, y * c.height / 4, c.width / 3, c.height / 4);
            cv::Scalar a, b;
            cv::meanStdDev(s.enhanced(r), a, b);
            out.push_back(float(a[0] / 255));
            out.push_back(float(b[0] / 255));
        }
    if (!c.residualFeatures)
        std::fill(texture.begin(), texture.end(), 0.f);
    out.insert(out.end(), texture.begin(), texture.end());
    return cv::Mat(1, int(out.size()), CV_32F, out.data()).clone();
}
cv::Mat localize(const cv::Mat &image, const Config &c, std::vector<Region> *regions) {
    auto s = preprocess(image, c);
    cv::Mat dark, bright, response, binary;
    auto kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {c.morphologyKernel, c.morphologyKernel});
    cv::morphologyEx(s.denoised, dark, cv::MORPH_BLACKHAT, kernel);
    cv::morphologyEx(s.denoised, bright, cv::MORPH_TOPHAT, kernel);
    cv::max(dark, bright, response);
    cv::threshold(response, binary, c.localThreshold, 255, cv::THRESH_BINARY);
    // Gradient support removes isolated intensity noise; closing joins broken candidate edges.
    cv::Mat support;
    s.gradient.convertTo(support, CV_8U);
    cv::threshold(support, support, c.gradientSupport, 255, cv::THRESH_BINARY);
    cv::dilate(support, support, cv::Mat::ones(3, 3, CV_8U));
    cv::bitwise_and(binary, support, binary);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
    int m = c.margin;
    binary.rowRange(0, m).setTo(0);
    binary.rowRange(binary.rows - m, binary.rows).setTo(0);
    binary.colRange(0, m).setTo(0);
    binary.colRange(binary.cols - m, binary.cols).setTo(0);
    cv::Mat labels, stats, centroids, clean = cv::Mat::zeros(binary.size(), CV_8U);
    int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
    for (int i = 1; i < n; i++)
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= c.minArea)
            clean.setTo(255, labels == i);
    cv::Mat result;
    cv::resize(clean, result, image.size(), 0, 0, cv::INTER_NEAREST);
    if (regions) {
        regions->clear();
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (auto &contour : contours) {
            Region r;
            r.contour = contour;
            r.box = cv::boundingRect(contour);
            cv::Mat one = cv::Mat::zeros(result.size(), CV_8U);
            cv::drawContours(one, std::vector<std::vector<cv::Point>>{contour}, 0, 255, cv::FILLED);
            cv::bitwise_and(one, result, one);
            r.pixelArea = cv::countNonZero(one);
            regions->push_back(r);
        }
    }
    return result;
}
void Model::fitScale(const cv::Mat &x) {
    cv::reduce(x, mean, 0, cv::REDUCE_AVG);
    scale = cv::Mat::ones(1, x.cols, CV_32F);
    for (int j = 0; j < x.cols; j++) {
        cv::Scalar m, sd;
        cv::meanStdDev(x.col(j), m, sd);
        scale.at<float>(j) = std::max(float(sd[0]), 1e-5f);
    }
}
cv::Mat Model::normalize(const cv::Mat &x) const {
    if (x.cols != mean.cols || x.type() != CV_32F)
        throw std::runtime_error("Feature/configuration dimension mismatch");
    cv::Mat y = x.clone();
    for (int i = 0; i < y.rows; i++)
        for (int j = 0; j < y.cols; j++)
            y.at<float>(i, j) = (y.at<float>(i, j) - mean.at<float>(j)) / scale.at<float>(j);
    return y;
}
void Model::train(const cv::Mat &x, const cv::Mat &y, std::string kind, double p, int seed) {
    cv::setRNGSeed(seed);
    config.kind = kind;
    fitScale(x);
    auto z = normalize(x);
    double pos = cv::countNonZero(y), neg = y.rows - pos;
    float weight = float(std::min(10.0, neg / std::max(1.0, pos)));
    cv::Mat weights(2, 1, CV_32F);
    weights.at<float>(0) = 1;
    weights.at<float>(1) = weight;
    if (kind == "rf") {
        auto rf = cv::ml::RTrees::create();
        rf->setMaxDepth(int(p));
        rf->setMinSampleCount(3);
        rf->setMaxCategories(2);
        rf->setActiveVarCount(0);
        rf->setPriors(weights);
        rf->setCalculateVarImportance(true);
        rf->setTermCriteria({cv::TermCriteria::MAX_ITER, 200, 0});
        rf->train(z, cv::ml::ROW_SAMPLE, y);
        classifier = rf;
        config.orientation = 1;
    } else {
        auto svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(p);
        svm->setGamma(1.0 / x.cols);
        svm->setClassWeights(weights);
        svm->setTermCriteria({cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 3000, 1e-5});
        svm->train(z, cv::ml::ROW_SAMPLE, y);
        classifier = svm;
        cv::Mat raw;
        svm->predict(z, raw, cv::ml::StatModel::RAW_OUTPUT);
        double a = 0, b = 0;
        for (int i = 0; i < y.rows; i++)
            (y.at<int>(i) ? a : b) += raw.at<float>(i);
        config.orientation = (a / pos > b / neg) ? 1 : -1;
    }
}
std::vector<float> Model::scores(const cv::Mat &x) const {
    if (!classifier || !classifier->isTrained())
        throw std::runtime_error("No trained model");
    auto z = normalize(x);
    std::vector<float> out(z.rows);
    if (config.kind == "rf") {
        auto rf = classifier.dynamicCast<cv::ml::RTrees>();
        cv::Mat votes;
        rf->getVotes(z, votes, 0);
        int pc = -1;
        for (int j = 0; j < votes.cols; j++)
            if (votes.at<int>(0, j) == 1)
                pc = j;
        if (pc < 0)
            throw std::runtime_error("Forest lacks positive class");
        for (int i = 0; i < z.rows; i++) {
            int total = 0;
            for (int j = 0; j < votes.cols; j++)
                total += votes.at<int>(i + 1, j);
            out[i] = float(votes.at<int>(i + 1, pc)) / total;
        }
    } else {
        cv::Mat raw;
        classifier->predict(z, raw, cv::ml::StatModel::RAW_OUTPUT);
        for (int i = 0; i < z.rows; i++)
            out[i] = float(raw.at<float>(i) * config.orientation);
    }
    return out;
}
Result Model::analyze(const cv::Mat &image) const {
    Result r;
    auto t = std::chrono::steady_clock::now();
    auto f = features(image, config);
    r.featureMs = milliseconds(t);
    t = std::chrono::steady_clock::now();
    r.score = scores(f)[0];
    r.defective = r.score >= config.threshold;
    r.predictMs = milliseconds(t);
    t = std::chrono::steady_clock::now();
    r.mask = localize(image, config, &r.regions);
    r.processed = preprocess(image, config).processed;
    cv::resize(r.processed, r.processed, image.size(), 0, 0, cv::INTER_LINEAR);
    r.localizeMs = milliseconds(t);
    return r;
}
void Model::save(const fs::path &dir) const {
    fs::create_directories(dir);
    classifier->save((dir / "classifier.xml").string());
    cv::FileStorage f((dir / "pipeline.yml").string(), cv::FileStorage::WRITE);
    config.write(f);
    f << "feature_count" << mean.cols << "mean" << mean << "scale" << scale;
}
void Model::load(const fs::path &dir) {
    cv::FileStorage f((dir / "pipeline.yml").string(), cv::FileStorage::READ);
    if (!f.isOpened())
        throw std::runtime_error("Missing saved pipeline: " + dir.string());
    config.read(f["configuration"]);
    f["mean"] >> mean;
    f["scale"] >> scale;
    if (mean.empty() || mean.size() != scale.size())
        throw std::runtime_error("Invalid saved scaling");
    if (config.kind == "rf")
        classifier = cv::Algorithm::load<cv::ml::RTrees>((dir / "classifier.xml").string());
    else if (config.kind == "svm")
        classifier = cv::Algorithm::load<cv::ml::SVM>((dir / "classifier.xml").string());
    else
        throw std::runtime_error("Unknown model type");
}
} // namespace ai
