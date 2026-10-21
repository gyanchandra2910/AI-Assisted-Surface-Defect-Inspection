#include "pipeline.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif
namespace ai {
static std::string pixelHash(const cv::Mat &image) {
    cv::Mat b = image.isContinuous() ? image : image.clone();
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("SHA256 unavailable");
    DWORD length = 0, got = 0;
    BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&length), sizeof(length),
                      &got, 0);
    std::vector<UCHAR> obj(length), digest(32);
    if (BCryptCreateHash(alg, &hash, obj.data(), length, nullptr, 0, 0) < 0)
        throw std::runtime_error("Hash initialization failed");
    int dims[3] = {b.rows, b.cols, b.type()};
    BCryptHashData(hash, reinterpret_cast<PUCHAR>(dims), sizeof(dims), 0);
    BCryptHashData(hash, b.data, ULONG(b.total() * b.elemSize()), 0);
    BCryptFinishHash(hash, digest.data(), 32, 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    std::ostringstream s;
    for (auto v : digest)
        s << std::hex << std::setw(2) << std::setfill('0') << int(v);
    return s.str();
#else
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < b.total() * b.elemSize(); i++)
        h = (h ^ b.data[i]) * 1099511628211ULL;
    return std::to_string(b.rows) + "x" + std::to_string(b.cols) + ":" + std::to_string(h);
#endif
}
std::vector<Sample> manifest(const fs::path &data, const fs::path &out) {
    if (!fs::is_directory(data))
        throw std::runtime_error("Dataset directory missing: " + data.string());
    fs::create_directories(out);
    std::vector<Sample> samples;
    std::vector<std::string> auxiliary;
    std::regex imageName("^[0-9]+\\.png$");
    for (auto &e : fs::recursive_directory_iterator(data)) {
        if (!e.is_regular_file())
            continue;
        std::string name = e.path().filename().string(),
                    split = e.path().parent_path().filename().string();
        if ((split == "train" || split == "test") && std::regex_match(name, imageName)) {
            Sample s;
            s.image = e.path();
            s.mask = e.path().parent_path() / (e.path().stem().string() + "_GT.png");
            s.id = e.path().stem().string();
            s.split = split;
            if (!fs::exists(s.mask))
                throw std::runtime_error("Unpaired image: " + s.image.string());
            auto im = cv::imread(s.image.string(), cv::IMREAD_COLOR),
                 gt = cv::imread(s.mask.string(), cv::IMREAD_GRAYSCALE);
            if (im.empty() || gt.empty())
                throw std::runtime_error("Dataset decode failed: " + s.image.string());
            if (im.size() != gt.size())
                throw std::runtime_error("Mask dimensions differ: " + s.id);
            s.label = cv::countNonZero(gt) > 0;
            s.width = im.cols;
            s.height = im.rows;
            s.hash = pixelHash(im);
            samples.push_back(s);
        } else if (name.find("_GT.png") == std::string::npos)
            auxiliary.push_back(fs::relative(e.path(), data).generic_string());
        else if (!fs::exists(e.path().parent_path() / (name.substr(0, name.size() - 7) + ".png")))
            throw std::runtime_error("Orphan mask: " + name);
    }
    std::sort(samples.begin(), samples.end(), [](const Sample &a, const Sample &b) {
        return std::tie(a.split, a.id) < std::tie(b.split, b.id);
    });
    std::map<std::string, std::vector<int>> groups;
    int train = 0, test = 0, tp = 0, ep = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        auto &s = samples[i];
        groups[s.hash].push_back(int(i));
        if (s.split == "train") {
            train++;
            tp += s.label;
        } else {
            test++;
            ep += s.label;
        }
    }
    if (train != 2331 || test != 1004 || tp != 246 || ep != 110)
        throw std::runtime_error(
            "Canonical class counts differ from official dataset: " + std::to_string(train) + "/" +
            std::to_string(tp) + ", " + std::to_string(test) + "/" + std::to_string(ep));
    std::ofstream csv(out / "manifest.csv");
    csv << "split,id,image,mask,label,width,height,pixel_sha256\n";
    for (auto &s : samples)
        csv << s.split << ',' << s.id << ',' << fs::relative(s.image, data).generic_string() << ','
            << fs::relative(s.mask, data).generic_string() << ',' << s.label << ',' << s.width
            << ',' << s.height << ',' << s.hash << '\n';
    std::ofstream audit(out / "dataset_audit.txt");
    audit << "Canonical training: " << train << "; defective " << tp << "; normal " << train - tp
          << "\nCanonical test: " << test << "; defective " << ep << "; normal " << test - ep
          << "\nEvery pair decoded and dimensions validated. Labels = nonempty paired "
             "mask.\nDecoded BGR pixels plus dimensions hashed with SHA256.\nAuxiliary entries "
             "preserved; never executed/deserialized:\n";
    for (auto &a : auxiliary) {
        audit << a;
        if (a.find("(copy)") != std::string::npos) {
            auto im = cv::imread((data / a).string());
            if (!im.empty()) {
                auto h = pixelHash(im);
                if (groups.count(h)) {
                    audit << " [same pixels as ";
                    for (auto i : groups[h])
                        audit << samples[i].split << '/' << samples[i].id << ' ';
                    audit << ']';
                } else
                    audit << " [not identical to canonical pixels]";
            }
        }
        audit << '\n';
    }
    for (auto &g : groups)
        if (g.second.size() > 1) {
            audit << "Canonical duplicate group: ";
            for (auto i : g.second)
                audit << samples[i].split << '/' << samples[i].id << ' ';
            audit << '\n';
            int label = samples[g.second[0]].label;
            for (int i : g.second)
                if (samples[i].label != label)
                    throw std::runtime_error("Conflicting labels in duplicate group");
        }
    std::cout << "Validated dataset: " << train << " train (" << tp << " defects), " << test
              << " test (" << ep << " defects)\n"
              << std::flush;
    return samples;
}
struct Metrics {
    int tn = 0, fp = 0, fn = 0, tp = 0;
    double precision = 0, recall = 0, f1 = 0, f2 = 0, accuracy = 0, auc = 0, ap = 0;
};
static Metrics metrics(const std::vector<int> &y, const std::vector<float> &scores,
                       double threshold) {
    Metrics m;
    int positives = 0;
    std::vector<int> order(y.size());
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < y.size(); i++) {
        bool p = scores[i] >= threshold;
        if (y[i]) {
            positives++;
            if (p)
                m.tp++;
            else
                m.fn++;
        } else {
            if (p)
                m.fp++;
            else
                m.tn++;
        }
    }
    m.precision = double(m.tp) / std::max(1, m.tp + m.fp);
    m.recall = double(m.tp) / std::max(1, m.tp + m.fn);
    m.f1 = 2 * m.precision * m.recall / std::max(1e-12, m.precision + m.recall);
    m.f2 = 5 * m.precision * m.recall / std::max(1e-12, 4 * m.precision + m.recall);
    m.accuracy = double(m.tp + m.tn) / y.size();
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    double tp = 0, fp = 0, prevTP = 0, prevFP = 0;
    size_t i = 0;
    while (i < order.size()) {
        size_t j = i;
        while (j < order.size() && scores[order[j]] == scores[order[i]]) {
            if (y[order[j]])
                tp++;
            else
                fp++;
            j++;
        }
        m.ap += (tp - prevTP) / std::max(1, positives) * tp / (tp + fp);
        m.auc += (fp - prevFP) * (tp + prevTP) / 2;
        prevTP = tp;
        prevFP = fp;
        i = j;
    }
    m.auc /= std::max(1, positives) * std::max(1, int(y.size()) - positives);
    return m;
}
static double chooseThreshold(const std::vector<int> &y, const std::vector<float> &scores) {
    auto s = scores;
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    double best = -1, threshold = s.front();
    for (float t : s) {
        auto m = metrics(y, scores, t);
        double objective = m.f2 + 1e-6 * m.f1;
        if (objective > best) {
            best = objective;
            threshold = t;
        }
    }
    return threshold;
}
static void writeMetrics(cv::FileStorage &f, const std::string &key, const Metrics &m) {
    f << key << "{" << "tn" << m.tn << "fp" << m.fp << "fn" << m.fn << "tp" << m.tp << "accuracy"
      << m.accuracy << "defect_precision" << m.precision << "defect_recall" << m.recall
      << "defect_f1" << m.f1 << "defect_f2" << m.f2 << "roc_auc" << m.auc << "average_precision"
      << m.ap << "}";
}
static cv::Mat getRows(const cv::Mat &x, const std::vector<int> &indices) {
    cv::Mat out;
    for (int i : indices)
        out.push_back(x.row(i));
    return out;
}
struct Localization {
    double meanIoU = 0, meanDice = 0, pixelPrecision = 0, pixelRecall = 0, normalArea = 0, ms = 0;
    int positives = 0, normals = 0;
};
static Localization localization(const std::vector<Sample> &samples,
                                 const std::vector<int> &indices, const Config &c,
                                 std::ofstream *perImage = nullptr) {
    Localization m;
    double overlap = 0, truthArea = 0, predictedArea = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i : indices) {
        auto &s = samples[i];
        auto image = cv::imread(s.image.string()), gt = cv::imread(s.mask.string(), 0);
        cv::threshold(gt, gt, 0, 255, cv::THRESH_BINARY);
        auto pred = localize(image, c);
        cv::Mat common;
        cv::bitwise_and(gt, pred, common);
        double inter = cv::countNonZero(common), g = cv::countNonZero(gt),
               p = cv::countNonZero(pred);
        double iou = inter / std::max(1.0, g + p - inter), dice = 2 * inter / std::max(1.0, g + p);
        if (s.label) {
            m.positives++;
            m.meanIoU += iou;
            m.meanDice += dice;
            overlap += inter;
            truthArea += g;
            predictedArea += p;
        } else {
            m.normals++;
            m.normalArea += p / pred.total();
        }
        if (perImage)
            *perImage << s.id << ',' << s.label << ',' << iou << ',' << dice << ','
                      << p / pred.total() << '\n';
    }
    m.meanIoU /= std::max(1, m.positives);
    m.meanDice /= std::max(1, m.positives);
    m.normalArea /= std::max(1, m.normals);
    m.pixelPrecision = overlap / std::max(1.0, predictedArea);
    m.pixelRecall = overlap / std::max(1.0, truthArea);
    m.ms = milliseconds(start) / indices.size();
    return m;
}
static void writeLocalization(cv::FileStorage &f, const Localization &m) {
    f << "localization" << "{" << "defective_images" << m.positives << "mean_defective_iou"
      << m.meanIoU << "mean_defective_dice" << m.meanDice << "defective_pixel_precision"
      << m.pixelPrecision << "defective_pixel_recall" << m.pixelRecall << "normal_images"
      << m.normals << "normal_false_positive_area_fraction" << m.normalArea
      << "localization_ms_per_image" << m.ms << "}";
}
static void representative(const Sample &s, const Model &model, const fs::path &dir,
                           const std::string &type) {
    auto image = cv::imread(s.image.string());
    auto r = model.analyze(image);
    cv::Mat gt = cv::imread(s.mask.string(), 0), truth = image.clone(), pred = image.clone();
    truth.setTo(cv::Scalar(255, 180, 20), gt);
    pred.setTo(cv::Scalar(20, 140, 255), r.mask);
    cv::addWeighted(image, 0.45, truth, 0.55, 0, truth);
    cv::addWeighted(image, 0.45, pred, 0.55, 0, pred);
    for (auto &a : r.regions)
        cv::rectangle(pred, a.box, {0, 220, 255}, 1);
    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{image, truth, pred}, combined);
    cv::copyMakeBorder(combined, combined, 30, 0, 0, 0, cv::BORDER_CONSTANT, {25, 25, 25});
    cv::putText(combined, "Original | Ground truth | Candidates (separate from classifier)",
                {6, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.38, {255, 255, 255}, 1);
    cv::imwrite((dir / (type + "_" + s.id + ".png")).string(), combined);
}
static void finalTest(const std::vector<Sample> &samples, const Model &model, const fs::path &out) {
    fs::create_directories(out / "examples");
    std::vector<int> indices, y;
    cv::Mat x;
    double featTime = 0, predictTime = 0;
    for (size_t i = 0; i < samples.size(); i++)
        if (samples[i].split == "test") {
            indices.push_back(int(i));
            y.push_back(samples[i].label);
            auto image = cv::imread(samples[i].image.string());
            auto t = std::chrono::steady_clock::now();
            x.push_back(features(image, model.config));
            featTime += milliseconds(t);
            if (indices.size() % 200 == 0)
                std::cout << "Test features " << indices.size() << "/1004\n" << std::flush;
        }
    auto t = std::chrono::steady_clock::now();
    auto scores = model.scores(x);
    predictTime = milliseconds(t);
    auto m = metrics(y, scores, model.config.threshold);
    auto baseline = metrics(y, std::vector<float>(y.size(), 0), 1);
    std::ofstream locCsv(out / "test_localization.csv");
    locCsv << "id,label,iou,dice,predicted_area_fraction\n";
    auto loc = localization(samples, indices, model.config, &locCsv);
    cv::FileStorage f((out / "final_test.json").string(),
                      cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    f << "test_images" << int(y.size()) << "model" << model.config.kind << "feature_count" << x.cols
      << "threshold" << model.config.threshold;
    writeMetrics(f, "chosen_model", m);
    writeMetrics(f, "majority_baseline", baseline);
    writeLocalization(f, loc);
    f << "timings" << "{" << "feature_ms_per_image" << featTime / y.size()
      << "batch_prediction_ms_per_image" << predictTime / y.size()
      << "feature_and_prediction_images_per_second" << 1000 * y.size() / (featTime + predictTime)
      << "}";
    std::ofstream csv(out / "test_predictions.csv");
    csv << "id,label,score,prediction,outcome\n";
    std::map<std::string, int> examples;
    for (size_t j = 0; j < y.size(); j++) {
        bool p = scores[j] >= model.config.threshold;
        std::string outcome = y[j] ? (p ? "TP" : "FN") : (p ? "FP" : "TN");
        csv << samples[indices[j]].id << ',' << y[j] << ',' << scores[j] << ',' << p << ','
            << outcome << '\n';
        if (examples[outcome]++ < 2)
            representative(samples[indices[j]], model, out / "examples", outcome);
    }
    std::ofstream cm(out / "confusion_matrix.csv");
    cm << "actual/predicted,normal,defective\nnormal," << m.tn << ',' << m.fp << "\ndefective,"
       << m.fn << ',' << m.tp << '\n';
    std::cout << "FINAL test: precision " << m.precision << ", recall " << m.recall << ", F1 "
              << m.f1 << ", AUC " << m.auc << "; TN " << m.tn << " FP " << m.fp << " FN " << m.fn
              << " TP " << m.tp << "\nFeature " << featTime / y.size() << " ms/image; prediction "
              << predictTime / y.size() << " ms/image; localization " << loc.ms
              << " ms/image; defective Dice " << loc.meanDice << ", normal FP area "
              << loc.normalArea << "\n"
              << std::flush;
}
void trainEvaluate(const fs::path &data, const fs::path &out) {
    cv::setNumThreads(1);
    auto start = std::chrono::steady_clock::now();
    auto samples = manifest(data, out);
    std::vector<int> development, test, fit, val;
    std::set<std::string> testHashes;
    for (auto &s : samples)
        if (s.split == "test")
            testHashes.insert(s.hash);
    std::map<std::string, std::vector<int>> groups;
    for (size_t i = 0; i < samples.size(); i++)
        if (samples[i].split == "train" && !testHashes.count(samples[i].hash))
            groups[samples[i].hash].push_back(int(i));
    std::vector<std::vector<int>> classes[2];
    for (auto &g : groups)
        classes[samples[g.second[0]].label].push_back(g.second);
    std::mt19937 rng(20261013);
    for (auto &cls : classes) {
        std::shuffle(cls.begin(), cls.end(), rng);
        size_t n = std::max(size_t(1), size_t(std::round(cls.size() * 0.2)));
        for (size_t j = 0; j < cls.size(); j++)
            for (int i : cls[j]) {
                development.push_back(i);
                (j < n ? val : fit).push_back(int(development.size() - 1));
            }
    }
    std::set<std::string> fitHashes;
    for (int i : fit)
        fitHashes.insert(samples[development[i]].hash);
    for (int i : val)
        if (fitHashes.count(samples[development[i]].hash))
            throw std::runtime_error("Duplicate crossed validation boundary");
    cv::Mat x, y;
    std::vector<int> vy;
    std::ofstream splits(out / "development_split.csv");
    splits << "id,subset,pixel_sha256\n";
    std::set<int> valSet(val.begin(), val.end());
    double featureMs = 0;
    Config base;
    for (size_t j = 0; j < development.size(); j++) {
        auto &s = samples[development[j]];
        auto image = cv::imread(s.image.string());
        auto t = std::chrono::steady_clock::now();
        x.push_back(features(image, base));
        featureMs += milliseconds(t);
        y.push_back(s.label);
        splits << s.id << ',' << (valSet.count(int(j)) ? "validation" : "fit") << ',' << s.hash
               << '\n';
        if ((j + 1) % 200 == 0)
            std::cout << "Development features " << j + 1 << '/' << development.size() << "\n"
                      << std::flush;
    }
    for (int j : val)
        vy.push_back(y.at<int>(j));
    std::cout << "Fit " << fit.size() << " validation " << val.size() << "; " << x.cols
              << " image-only features\n"
              << std::flush;
    auto trainX = getRows(x, fit), trainY = getRows(y, fit), valX = getRows(x, val);
    Model best;
    double bestObjective = -1;
    std::string chosen;
    std::ofstream comp(out / "model_comparison.csv");
    comp << "variant,model,parameter,threshold,precision,recall,f1,f2,roc_auc,average_precision,fp,"
            "fn,fit_ms\n";
    auto b = metrics(vy, std::vector<float>(vy.size(), 0), 1);
    comp << "majority,constant,0,1," << b.precision << ',' << b.recall << ',' << b.f1 << ',' << b.f2
         << ',' << b.auc << ',' << b.ap << ',' << b.fp << ',' << b.fn << ",0\n";
    for (bool residual : {true, false}) {
        cv::Mat tx = trainX.clone(), vx = valX.clone();
        if (!residual) {
            tx.colRange(51, tx.cols).setTo(0);
            vx.colRange(51, vx.cols).setTo(0);
        }
        for (auto candidate : std::vector<std::pair<std::string, double>>{
                 {"rf", 10}, {"rf", 16}, {"svm", 1}, {"svm", 10}}) {
            Model model;
            model.config = base;
            model.config.residualFeatures = residual;
            auto t = std::chrono::steady_clock::now();
            model.train(tx, trainY, candidate.first, candidate.second, 20261013);
            double fitMs = milliseconds(t);
            auto scores = model.scores(vx);
            model.config.threshold = chooseThreshold(vy, scores);
            auto m = metrics(vy, scores, model.config.threshold);
            std::string variant = residual ? "full" : "without_residual_texture";
            comp << variant << ',' << candidate.first << ',' << candidate.second << ','
                 << model.config.threshold << ',' << m.precision << ',' << m.recall << ',' << m.f1
                 << ',' << m.f2 << ',' << m.auc << ',' << m.ap << ',' << m.fp << ',' << m.fn << ','
                 << fitMs << '\n';
            comp.flush();
            std::cout << variant << ' ' << candidate.first << ' ' << candidate.second
                      << ": validation P=" << m.precision << " R=" << m.recall << " F1=" << m.f1
                      << " F2=" << m.f2 << " AUC=" << m.auc << "\n"
                      << std::flush;
            double objective = m.f2 + 1e-6 * m.f1;
            if (objective > bestObjective) {
                bestObjective = objective;
                best = model;
                chosen = variant + " " + candidate.first + " " + std::to_string(candidate.second);
            }
        }
    }
    std::vector<int> valSamples;
    for (int i : val)
        valSamples.push_back(development[i]);
    std::ofstream loc(out / "localization_validation.csv");
    loc << "threshold,min_area,defective_iou,defective_dice,defective_pixel_precision,defective_"
           "pixel_recall,normal_fp_area,objective\n";
    double bestLoc = -1;
    for (int threshold : {12, 20, 30, 45})
        for (int area : {8, 24}) {
            Config c = best.config;
            c.localThreshold = threshold;
            c.minArea = area;
            auto m = localization(samples, valSamples, c);
            double objective = m.meanDice - 2 * m.normalArea;
            loc << threshold << ',' << area << ',' << m.meanIoU << ',' << m.meanDice << ','
                << m.pixelPrecision << ',' << m.pixelRecall << ',' << m.normalArea << ','
                << objective << '\n';
            loc.flush();
            if (objective > bestLoc) {
                bestLoc = objective;
                best.config = c;
            }
            std::cout << "Localization validation t=" << threshold << " area=" << area
                      << " Dice=" << m.meanDice << " normal FP=" << m.normalArea << "\n"
                      << std::flush;
        }
    best.save(out / "model");
    Model loaded;
    loaded.load(out / "model");
    auto chosenX = valX.clone();
    if (!best.config.residualFeatures)
        chosenX.colRange(51, chosenX.cols).setTo(0);
    auto before = best.scores(chosenX), after = loaded.scores(chosenX);
    double delta = 0;
    for (size_t i = 0; i < before.size(); i++)
        delta = std::max(delta, double(std::abs(before[i] - after[i])));
    if (delta > 1e-6)
        throw std::runtime_error("Saved model changed predictions");
    // Also exercise actual feature construction from images with the persisted configuration.
    for (int k = 0; k < std::min(12, int(valSamples.size())); k++) {
        auto im = cv::imread(samples[valSamples[k]].image.string());
        auto f = features(im, loaded.config);
        if (cv::norm(f, chosenX.row(k), cv::NORM_INF) > 1e-6)
            throw std::runtime_error("Training/inference feature mismatch");
    }
    cv::FileStorage summary((out / "training_summary.json").string(),
                            cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    summary << "seed" << 20261013 << "fit_images" << int(fit.size()) << "validation_images"
            << int(val.size()) << "cross_test_duplicate_train_images_excluded"
            << 2331 - int(development.size()) << "selected" << chosen << "selection_objective"
            << "validation F2, then F1" << "saved_reload_max_score_difference" << delta
            << "feature_ms_per_image" << featureMs / development.size()
            << "development_total_seconds" << milliseconds(start) / 1000
            << "model_refit_on_validation" << 0;
    writeMetrics(summary, "chosen_validation", metrics(vy, after, best.config.threshold));
    std::cout << "Pipeline locked; evaluating untouched official test split\n" << std::flush;
    finalTest(samples, loaded, out);
}
void evaluateSaved(const fs::path &data, const fs::path &model, const fs::path &out) {
    cv::setNumThreads(1);
    Model m;
    m.load(model);
    auto s = manifest(data, out);
    finalTest(s, m, out);
}
} // namespace ai
