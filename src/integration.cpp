#include "integration.hpp"
#include <fstream>
#include <iostream>
#include <regex>
namespace app {
Processor::Processor(const ai::fs::path &dir) {
    model.load(dir);
    worker = std::thread(&Processor::run, this);
}
Processor::~Processor() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    condition.notify_one();
    worker.join();
}
uint64_t Processor::request(int type, const std::string &path, int frame) {
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t token = ++requested;
    pending = Job{token, type, frame, path};
    completed.reset();
    condition.notify_one();
    return token;
}
std::optional<Packet> Processor::take() {
    std::lock_guard<std::mutex> lock(mutex);
    if (completed && gfx::matchesSource(requested, completed->token)) {
        auto result = std::move(completed);
        completed.reset();
        return result;
    }
    return {};
}
void Processor::run() {
    cv::VideoCapture cap;
    std::string videoSource;
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&] { return stopping || pending.has_value(); });
            if (stopping)
                return;
            job = *pending;
            pending.reset();
        }
        Packet packet;
        packet.token = job.token;
        try {
            auto start = std::chrono::steady_clock::now();
            if (job.type == 0) {
                cap.release();
                packet.source = job.path;
                packet.image = cv::imread(job.path, cv::IMREAD_COLOR);
                if (packet.image.empty())
                    throw std::runtime_error("Cannot decode image: " + job.path);
            } else {
                if (job.type == 1) {
                    cap.release();
                    if (!cap.open(job.path, cv::CAP_FFMPEG))
                        throw std::runtime_error("Cannot open video: " + job.path);
                    videoSource = job.path;
                }
                if (!cap.isOpened())
                    throw std::runtime_error("No video is open");
                if (job.type == 3 && !cap.set(cv::CAP_PROP_POS_FRAMES, job.frame))
                    throw std::runtime_error("Video seek failed");
                packet.source = videoSource;
                packet.frame = int(std::round(cap.get(cv::CAP_PROP_POS_FRAMES)));
                packet.totalFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
                packet.fps = cap.get(cv::CAP_PROP_FPS);
                if (!cap.read(packet.image) || packet.image.empty())
                    throw std::runtime_error("End of video or frame decoding failed");
            }
            packet.decodeMs = ai::milliseconds(start);
            packet.result = model.analyze(packet.image);
        } catch (const std::exception &e) {
            packet.error = e.what();
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (gfx::matchesSource(requested, packet.token))
                completed = std::move(packet);
        }
    }
}
static std::vector<ai::fs::path> canonicalImages(const ai::fs::path &data) {
    std::vector<ai::fs::path> p;
    std::regex rx("^[0-9]+\\.png$");
    for (auto &e : ai::fs::recursive_directory_iterator(data))
        if (e.is_regular_file() && e.path().parent_path().filename() == "test" &&
            std::regex_match(e.path().filename().string(), rx))
            p.push_back(e.path());
    std::sort(p.begin(), p.end());
    return p;
}
void makeDemo(const ai::fs::path &data, const ai::fs::path &output) {
    auto images = canonicalImages(data);
    if (images.empty())
        throw std::runtime_error("No canonical test images");
    ai::fs::create_directories(output.parent_path());
    auto first = cv::imread(images[0].string());
    cv::VideoWriter writer(output.string(), cv::CAP_FFMPEG,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 8, first.size());
    if (!writer.isOpened())
        throw std::runtime_error("Cannot create MJPEG demonstration video");
    std::ofstream listing(output.parent_path() / "demo_frames.csv");
    listing << "frame,source_image\n";
    for (int i = 0; i < std::min(24, int(images.size())); i++) {
        auto im = cv::imread(images[i].string());
        cv::resize(im, im, first.size());
        writer.write(im);
        listing << i << ',' << images[i].filename().string() << '\n';
    }
    std::cout << "Created dataset image-sequence demonstration: " << output
              << " (not independent video evaluation)\n";
}
static Packet wait(Processor &p, uint64_t token) {
    auto start = std::chrono::steady_clock::now();
    while (ai::milliseconds(start) < 15000) {
        auto result = p.take();
        if (result) {
            if (result->token != token)
                throw std::runtime_error("Stale processing result accepted");
            return std::move(*result);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("Analysis worker timeout");
}
void verifyRuntime(const ai::fs::path &data, const ai::fs::path &dir, const ai::fs::path &video,
                   const ai::fs::path &output) {
    auto images = canonicalImages(data);
    if (images.size() < 2)
        throw std::runtime_error("Missing test images");
    Processor p(dir);
    auto invalid = wait(p, p.request(0, "missing_image.png"));
    if (invalid.error.empty())
        throw std::runtime_error("Missing input was not rejected");
    ai::fs::create_directories(output);
    auto corruptPath = output / "corrupt_decode_check.png";
    {
        std::ofstream corrupt(corruptPath);
        corrupt << "invalid PNG fixture";
    }
    auto corrupt = wait(p, p.request(0, corruptPath.string()));
    ai::fs::remove(corruptPath);
    if (corrupt.error.empty())
        throw std::runtime_error("Corrupt image decoding was not rejected");
    p.request(0, images[0].string());
    auto latest = wait(p, p.request(0, images[1].string()));
    if (!latest.error.empty() || latest.source != images[1].string())
        throw std::runtime_error("Latest request isolation failed");
    auto image = cv::imread(images[1].string());
    ai::Model model;
    model.load(dir);
    auto direct = model.analyze(image);
    if (std::abs(direct.score - latest.result.score) > 1e-6 ||
        cv::norm(image, latest.image, cv::NORM_INF) > 0)
        throw std::runtime_error("Worker feature/source mismatch");
    // Copy only the image into an annotation-free directory, then predict using only image + saved
    // model.
    ai::fs::create_directories(output);
    auto copy = output / "inference_only.png";
    cv::imwrite(copy.string(), image);
    auto plain = wait(p, p.request(0, copy.string()));
    ai::fs::remove(copy);
    if (std::abs(direct.score - plain.result.score) > 1e-6)
        throw std::runtime_error("Image-only inference mismatch");
    cv::VideoCapture reference(video.string(), cv::CAP_FFMPEG);
    if (!reference.isOpened())
        throw std::runtime_error("Video reference decode failed");
    double maxScoreDelta = 0;
    int checked = 0;
    for (int i = 0; i < 6; i++) {
        auto packet = wait(p, p.request(i == 0 ? 1 : 2, i == 0 ? video.string() : ""));
        cv::Mat expected;
        reference.read(expected);
        if (!packet.error.empty() || packet.frame != i ||
            cv::norm(packet.image, expected, cv::NORM_INF) > 0)
            throw std::runtime_error("Video frame/result synchronization failed");
        auto expectedResult = model.analyze(expected);
        maxScoreDelta =
            std::max(maxScoreDelta, double(std::abs(packet.result.score - expectedResult.score)));
        checked++;
    }
    auto seek = wait(p, p.request(3, "", 2));
    reference.set(cv::CAP_PROP_POS_FRAMES, 2);
    cv::Mat expected;
    reference.read(expected);
    if (seek.frame != 2 || cv::norm(expected, seek.image, cv::NORM_INF) > 0)
        throw std::runtime_error("Video seek synchronization failed");
    if (maxScoreDelta > 1e-6)
        throw std::runtime_error("Video result attached to wrong frame");
    cv::FileStorage f((output / "runtime_checks.json").string(),
                      cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    f << "missing_input_rejected" << 1 << "failed_decode_rejected" << 1 << "latest_request_wins"
      << 1 << "image_only_saved_model_inference" << 1 << "video_frames_checked" << checked
      << "video_seek_checked" << 1 << "max_video_score_difference" << maxScoreDelta
      << "queue_capacity" << 1;
    std::cout << "Runtime checks passed: missing input, image-only inference, latest request, 6 "
                 "real decoded video frames and seek\n";
}
} // namespace app
