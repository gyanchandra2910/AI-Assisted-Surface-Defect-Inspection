#include "geometry.hpp"
#include "pipeline.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
static void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static bool near(gfx::Point a, gfx::Point b) {
    return std::hypot(a.x - b.x, a.y - b.y) < 1e-7;
}
int main() {
    try {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> u(0, 1);
        for (int i = 0; i < 500; i++) {
            double w = 230, h = 630;
            gfx::Rect viewport{340, 20, 1100 + u(rng) * 500, 700 + u(rng) * 200};
            double zoom = .2 + u(rng) * 5, angle = u(rng) * 6.283185;
            auto m = gfx::imageTransform(w, h, viewport, zoom, angle,
                                         {u(rng) * 300 - 150, u(rng) * 300 - 150});
            gfx::Point p{u(rng) * w, u(rng) * h};
            require(near(m.inverse().apply(m.apply(p)), p),
                    "Transform inverse failed after rotation/zoom/resize");
            std::vector<gfx::Point> polygon{{40, 100}, {110, 100}, {110, 200}, {40, 200}};
            require(gfx::pointInPolygon(m.inverse().apply(m.apply({60, 150})), polygon),
                    "Transformed picking lost alignment");
            require(!gfx::pointInPolygon(m.inverse().apply(m.apply({150, 250})), polygon),
                    "Picking accepted outside point");
        }
        gfx::Rect r{0, 0, 10, 10};
        gfx::Point a{-5, 5}, b{15, 5};
        require(gfx::liangBarsky(r, a, b) && near(a, {0, 5}) && near(b, {10, 5}),
                "Horizontal clipping failed");
        a = {-1, 1};
        b = {-1, 9};
        require(!gfx::liangBarsky(r, a, b), "Parallel outside accepted");
        a = {0, -2};
        b = {0, 12};
        require(gfx::liangBarsky(r, a, b) && near(a, {0, 0}) && near(b, {0, 10}),
                "Boundary parallel clipping failed");
        a = {5, 5};
        b = a;
        require(gfx::liangBarsky(r, a, b), "Interior zero-length rejected");
        a = {-1, -1};
        b = a;
        require(!gfx::liangBarsky(r, a, b), "Exterior zero-length accepted");
        a = {-3, -3};
        b = {0, 0};
        require(gfx::liangBarsky(r, a, b) && near(a, {0, 0}) && near(b, {0, 0}),
                "Corner tangent clipping failed");
        for (int i = 0; i < 500; i++) {
            a = {u(rng) * 40 - 15, u(rng) * 40 - 15};
            b = {u(rng) * 40 - 15, u(rng) * 40 - 15};
            auto c = a, d = b;
            bool x = gfx::liangBarsky(r, a, b), y = gfx::liangBarsky(r, d, c);
            require(x == y, "Clipping depends on direction");
            if (x) {
                require(near(a, c) && near(b, d), "Reversed clipping endpoints differ");
                require(a.x >= -1e-8 && a.x <= 10 + 1e-8 && a.y >= -1e-8 && a.y <= 10 + 1e-8,
                        "Clipped point outside ROI");
            }
        }
        require(gfx::matchesSource(8, 8) && !gfx::matchesSource(9, 8), "Stale frame tag accepted");
        bool rejected = false;
        try {
            ai::features(cv::Mat(), ai::Config{});
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Missing decoded input accepted");
        // Small deterministic fixtures test channel/order/configuration behavior, never used as
        // training data.
        cv::Mat gray(96, 48, CV_8U);
        for (int y = 0; y < gray.rows; y++)
            for (int x = 0; x < gray.cols; x++)
                gray.at<unsigned char>(y, x) = static_cast<unsigned char>((x * 3 + y) % 256);
        cv::Mat color;
        cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
        ai::Config config;
        auto f = ai::features(gray, config), g = ai::features(color, config);
        require(f.cols == 110 && cv::checkRange(f), "Invalid bounded feature schema");
        require(cv::norm(f, g, cv::NORM_INF) < 1e-6, "Grayscale/BGR feature inconsistency");
        config.residualFeatures = false;
        auto ablated = ai::features(color, config);
        require(cv::norm(ablated.colRange(0, 51), f.colRange(0, 51), cv::NORM_INF) < 1e-6 &&
                    cv::countNonZero(ablated.colRange(51, 110)) == 0,
                "Feature ablation affects wrong feature columns");
        ai::Model scaler;
        cv::Mat fit;
        fit.push_back(f);
        fit.push_back(f * 1.1f);
        scaler.fitScale(fit);
        auto normalized = scaler.normalize(fit);
        require(cv::checkRange(normalized), "Scaling produces invalid values");
        rejected = false;
        try {
            scaler.normalize(cv::Mat::zeros(1, 3, CV_32F));
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Feature dimension mismatch accepted");
        auto mask = ai::localize(color, ai::Config{});
        require(mask.size() == color.size() && mask.type() == CV_8U,
                "Localization dimensions differ from source");
        std::cout << "PASS: 500 inverse/picking cases, clipping "
                     "parallel/boundary/tangent/degenerate + 500 reversals, stale tags, decoding, "
                     "feature schema/ablation/scaling, localization dimensions\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "TEST FAILED: " << e.what() << '\n';
        return 1;
    }
}
