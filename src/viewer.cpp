#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "integration.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif
namespace app {
static std::string openFile(bool video) {
#ifdef _WIN32
    char name[4096] = {};
    OPENFILENAMEA dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.lpstrFile = name;
    dialog.nMaxFile = sizeof(name);
    dialog.lpstrFilter = video ? "Video\0*.avi;*.mp4;*.mov;*.mkv\0All files\0*.*\0"
                               : "Images\0*.png;*.jpg;*.jpeg;*.bmp;*.tif\0All files\0*.*\0";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&dialog) ? std::string(name) : "";
#else
    return "";
#endif
}
static GLuint compile(GLenum type, const char *source) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &source, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("Shader compilation: ") + log);
    }
    return s;
}
class Renderer {
    GLuint program = 0, vao = 0, vbo = 0, texture = 0;
    GLint transformLoc, sizeLoc, texturedLoc, colorLoc;
    gfx::Mat3 matrix;
    float width = 1, height = 1;
    void uniforms(bool textured, ImVec4 color) {
        glUseProgram(program);
        float a[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                a[j * 3 + i] = float(matrix.a[i * 3 + j]);
        glUniformMatrix3fv(transformLoc, 1, GL_FALSE, a);
        glUniform2f(sizeLoc, width, height);
        glUniform1i(texturedLoc, textured);
        glUniform4f(colorLoc, color.x, color.y, color.z, color.w);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }

  public:
    Renderer() {
        const char *vs = R"(#version 330 core
 layout(location=0) in vec2 position;layout(location=1) in vec2 texcoord;
 uniform mat3 transform;uniform vec2 displaySize;out vec2 uv;
 void main(){vec3 screen=transform*vec3(position,1);gl_Position=vec4(2*screen.x/displaySize.x-1,1-2*screen.y/displaySize.y,0,1);uv=texcoord;})";
        const char *fs = R"(#version 330 core
 in vec2 uv;uniform sampler2D imageTexture;uniform bool textured;uniform vec4 color;out vec4 frag;
 void main(){frag=textured?texture(imageTexture,uv):color;})";
        GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
        program = glCreateProgram();
        glAttachShader(program, v);
        glAttachShader(program, f);
        glLinkProgram(program);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok)
            throw std::runtime_error("Shader link failed");
        transformLoc = glGetUniformLocation(program, "transform");
        sizeLoc = glGetUniformLocation(program, "displaySize");
        texturedLoc = glGetUniformLocation(program, "textured");
        colorLoc = glGetUniformLocation(program, "color");
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "imageTexture"), 0);
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void *)(2 * sizeof(float)));
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    ~Renderer() {
        glDeleteTextures(1, &texture);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
    }
    void setTransform(gfx::Mat3 m, float w, float h) {
        matrix = m;
        width = w;
        height = h;
    }
    void upload(const cv::Mat &image) {
        cv::Mat rgb;
        if (image.channels() == 1)
            cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
        if (!rgb.isContinuous())
            rgb = rgb.clone();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     rgb.data);
    }
    void image(int w, int h) {
        float v[] = {0, 0, 0, 0, float(w), 0,        1, 0, float(w), float(h), 1, 1,
                     0, 0, 0, 0, float(w), float(h), 1, 1, 0,        float(h), 0, 1};
        uniforms(true, {1, 1, 1, 1});
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    void lines(const std::vector<gfx::Point> &points, ImVec4 color, float lineWidth) {
        if (points.empty())
            return;
        std::vector<float> v;
        for (auto p : points) {
            v.push_back(float(p.x));
            v.push_back(float(p.y));
            v.push_back(0);
            v.push_back(0);
        }
        uniforms(false, color);
        glLineWidth(lineWidth);
        glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_LINES, 0, int(points.size()));
    }
};
static std::vector<gfx::Point> segments(const std::vector<cv::Point> &polygon,
                                        const std::optional<gfx::Rect> &clip) {
    std::vector<gfx::Point> lines;
    for (size_t i = 0; i < polygon.size(); i++) {
        auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
        gfx::Point x{double(a.x), double(a.y)}, y{double(b.x), double(b.y)};
        if (!clip || gfx::liangBarsky(*clip, x, y)) {
            lines.push_back(x);
            lines.push_back(y);
        }
    }
    return lines;
}
static std::vector<cv::Point> boxPolygon(cv::Rect b) {
    return {
        {b.x, b.y}, {b.x + b.width, b.y}, {b.x + b.width, b.y + b.height}, {b.x, b.y + b.height}};
}
static int pick(const Packet &packet, gfx::Mat3 transform, gfx::Point screen,
                const std::optional<gfx::Rect> &roi) {
    auto p = transform.inverse().apply(screen);
    if (roi && !roi->contains(p))
        return -1;
    for (int i = int(packet.result.regions.size()) - 1; i >= 0; i--)
        if (cv::pointPolygonTest(packet.result.regions[i].contour, {float(p.x), float(p.y)},
                                 false) >= 0)
            return i;
    return -1;
}
static void screenshot(const ai::fs::path &file, int w, int h) {
    cv::Mat rgb(h, w, CV_8UC3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    cv::flip(rgb, rgb, 0);
    cv::cvtColor(rgb, rgb, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(file.string(), rgb))
        throw std::runtime_error("Screenshot save failed");
}
// Compare actual framebuffer samples with bilinear source pixels. This independently checks
// the shader matrix, RGB order, image origin, texture alignment and framebuffer scaling.
static int verifyTexture(const Packet &packet, gfx::Mat3 transform, int ww, int wh, int fw,
                         int fh) {
    int maxError = 0;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    for (double y : {0.25, 0.5, 0.75}) {
        for (double x : {0.25, 0.5, 0.75}) {
            auto screen = transform.apply({x * packet.image.cols, y * packet.image.rows});
            int px = int(std::round(screen.x * fw / ww - 0.5));
            int py = int(std::round(screen.y * fh / wh - 0.5));
            auto local = transform.inverse().apply({(px + 0.5) * ww / fw, (py + 0.5) * wh / fh});
            cv::Mat expected;
            cv::getRectSubPix(packet.image, {1, 1}, {float(local.x - 0.5), float(local.y - 0.5)},
                              expected);
            unsigned char rgb[3];
            glReadPixels(px, fh - 1 - py, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, rgb);
            auto bgr = expected.at<cv::Vec3b>(0, 0);
            for (int k = 0; k < 3; k++)
                maxError = std::max(maxError, std::abs(int(rgb[k]) - int(bgr[2 - k])));
        }
    }
    if (maxError > 12)
        throw std::runtime_error("Framebuffer/source texture alignment failed: " +
                                 std::to_string(maxError));
    return maxError;
}
int viewer(const ai::fs::path &data, const ai::fs::path &modelDir, const std::string &initial,
           bool smoke, const ai::fs::path &output) {
    Processor processor(modelDir);
    std::vector<ai::fs::path> images;
    std::regex canonical("^[0-9]+\\.png$");
    if (ai::fs::exists(data))
        for (auto &e : ai::fs::recursive_directory_iterator(data))
            if (e.is_regular_file() && std::regex_match(e.path().filename().string(), canonical))
                images.push_back(e.path());
    std::sort(images.begin(), images.end());
    std::string first = initial;
    if (first.empty()) {
        std::ifstream pred(output / "test_predictions.csv");
        std::string line;
        while (std::getline(pred, line)) {
            if (line.size() > 3 && line.substr(line.size() - 3) == ",TP") {
                std::string id = line.substr(0, line.find(','));
                for (auto &path : images)
                    if (path.stem() == id && path.parent_path().filename() == "test") {
                        first = path.string();
                        break;
                    }
                if (!first.empty())
                    break;
            }
        }
        if (first.empty() && !images.empty())
            first = images[0].string();
    }
    if (!glfwInit())
        throw std::runtime_error("GLFW initialization failed (desktop access required)");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(
        1280, 850, "AI-Assisted Surface Defect Detection and Interactive Visualization", nullptr,
        nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("OpenGL 3.3 context creation failed");
    }
    glfwSetWindowSizeLimits(window, 700, 500, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(smoke ? 0 : 1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("GLEW initialization failed");
    while (glGetError() != GL_NO_ERROR) {
    }
    std::string glVersion = reinterpret_cast<const char *>(glGetString(GL_VERSION)),
                glRenderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    std::cout << "OpenGL " << glVersion << "; renderer " << glRenderer << '\n' << std::flush;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    auto &style = ImGui::GetStyle();
    style.WindowRounding = 0;
    style.FrameRounding = 4;
    style.Colors[ImGuiCol_WindowBg] = {0.06f, 0.08f, 0.11f, 1};
    style.Colors[ImGuiCol_Button] = {0.14f, 0.25f, 0.33f, 1};
    style.Colors[ImGuiCol_Header] = {0.12f, 0.35f, 0.42f, 1};
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    int code = 0;
    {
        Renderer renderer;
        std::optional<Packet> displayed;
        uint64_t requested = 0;
        bool pending = false, playing = false, showBoxes = true, showGT = false, roiMode = false,
             clipROI = true, dragROI = false;
        int mode = 2, uploadedMode = -1, index = 0, selected = -1;
        float zoom = 1, angle = 0;
        gfx::Point pan{}, roiStart{};
        std::optional<gfx::Rect> roi;
        std::string error;
        double videoNext = 0, lastRenderMs = 0;
        std::vector<double> renderTimes;
        double analysisTotal = 0;
        int analyzed = 0, smokeFrames = 0, shots = 0, smokeVideoFrames = 0;
        int textureSampleError = -1;
        bool smokeVideo = false;
        std::string dropped;
        ai::fs::path groundTruthPath;
        bool groundTruthAvailable = false;
        uint64_t groundTruthToken = 0;
        std::vector<std::vector<cv::Point>> groundTruthContours;
        glfwSetWindowUserPointer(window, &dropped);
        glfwSetDropCallback(window, [](GLFWwindow *w, int count, const char **paths) {
            if (count)
                *static_cast<std::string *>(glfwGetWindowUserPointer(w)) = paths[0];
        });
        auto request = [&](int type, const std::string &path = "", int frame = 0) {
            requested = processor.request(type, path, frame);
            pending = true;
            error.clear();
            if (type == 0 || type == 1) {
                playing = false;
                selected = -1;
                roi.reset();
                zoom = 1;
                angle = 0;
                pan = {};
            }
            return requested;
        };
        if (!first.empty()) {
            request(0, first);
            auto found = std::find(images.begin(), images.end(), ai::fs::path(first));
            if (found != images.end())
                index = int(found - images.begin());
        }
        auto start = std::chrono::steady_clock::now();
        ai::fs::create_directories(output / "screenshots");
        try {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                if (!dropped.empty()) {
                    std::string path = std::move(dropped);
                    dropped.clear();
                    auto ext = ai::fs::path(path).extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c) { return char(std::tolower(c)); });
                    request(ext == ".avi" || ext == ".mp4" || ext == ".mkv" || ext == ".mov" ? 1
                                                                                             : 0,
                            path);
                }
                if (auto result = processor.take()) {
                    if (!gfx::matchesSource(requested, result->token))
                        throw std::runtime_error("Stale graphics result");
                    pending = false;
                    if (!result->error.empty()) {
                        error = result->error;
                        playing = false;
                    } else {
                        displayed = std::move(result);
                        uploadedMode = -1;
                        selected = -1;
                        groundTruthContours.clear();
                        groundTruthToken = 0;
                        ai::fs::path sourcePath = displayed->source;
                        groundTruthPath =
                            sourcePath.parent_path() / (sourcePath.stem().string() + "_GT.png");
                        groundTruthAvailable =
                            displayed->frame < 0 && ai::fs::is_regular_file(groundTruthPath);
                        if (!groundTruthAvailable)
                            showGT = false;
                        analysisTotal += displayed->result.featureMs + displayed->result.predictMs +
                                         displayed->result.localizeMs;
                        analyzed++;
                        if (smokeVideo) {
                            if (displayed->frame != smokeVideoFrames)
                                throw std::runtime_error("Smoke video source/frame mismatch");
                            smokeVideoFrames++;
                        }
                        videoNext = glfwGetTime() + 1 / std::max(1.0, displayed->fps);
                    }
                }
                int ww, wh, fw, fh;
                glfwGetWindowSize(window, &ww, &wh);
                glfwGetFramebufferSize(window, &fw, &fh);
                if (ww == 0 || wh == 0 || fw == 0 || fh == 0) {
                    glfwWaitEventsTimeout(0.05);
                    continue;
                }
                float sidebar = 340;
                gfx::Rect viewport{sidebar + 12, 12, double(ww) - 12, double(wh) - 12};
                gfx::Mat3 transform;
                if (displayed)
                    transform =
                        gfx::imageTransform(displayed->image.cols, displayed->image.rows, viewport,
                                            zoom, angle * 3.141592653589793 / 180, pan);
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                ImGui::SetNextWindowPos({0, 0});
                ImGui::SetNextWindowSize({sidebar, float(wh)});
                ImGui::Begin("Inspection controls", nullptr,
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);
                ImGui::TextColored({0.30f, 0.86f, 0.85f, 1}, "SURFACE / INSPECT");
                ImGui::TextWrapped("AI-Assisted Surface Defect Detection");
                ImGui::TextDisabled("Theme 6  |  IIITDM Kancheepuram");
                ImGui::Separator();
                if (ImGui::Button("Open image")) {
                    auto path = openFile(false);
                    if (!path.empty())
                        request(0, path);
                }
                ImGui::SameLine();
                if (ImGui::Button("Open video")) {
                    auto path = openFile(true);
                    if (!path.empty())
                        request(1, path);
                }
                if (!images.empty()) {
                    if (ImGui::Button("< Previous") && index > 0)
                        request(0, images[--index].string());
                    ImGui::SameLine();
                    if (ImGui::Button("Next >") && index + 1 < int(images.size()))
                        request(0, images[++index].string());
                    std::string current = images[index].parent_path().filename().string() + "/" +
                                          images[index].filename().string();
                    if (ImGui::BeginCombo("Dataset", current.c_str())) {
                        ImGuiListClipper clip;
                        clip.Begin(int(images.size()));
                        while (clip.Step())
                            for (int i = clip.DisplayStart; i < clip.DisplayEnd; i++) {
                                std::string label = images[i].parent_path().filename().string() +
                                                    "/" + images[i].filename().string();
                                if (ImGui::Selectable(label.c_str(), i == index)) {
                                    index = i;
                                    request(0, images[i].string());
                                }
                                if (i == index)
                                    ImGui::SetItemDefaultFocus();
                            }
                        ImGui::EndCombo();
                    }
                }
                if (pending)
                    ImGui::TextColored({1, 0.78f, 0.32f, 1}, "Analyzing requested source...");
                if (!error.empty())
                    ImGui::TextWrapped("%s", error.c_str());
                if (displayed) {
                    auto &p = *displayed;
                    ImGui::TextWrapped("Displayed: %s",
                                       ai::fs::path(p.source).filename().string().c_str());
                    ImGui::TextDisabled("%d x %d pixels  |  source token %llu", p.image.cols,
                                        p.image.rows, (unsigned long long)p.token);
                    ImGui::Separator();
                    ImGui::TextColored(p.result.defective ? ImVec4{1, 0.47f, 0.30f, 1}
                                                          : ImVec4{0.35f, 0.84f, 0.6f, 1},
                                       "Classifier: %s",
                                       p.result.defective ? "DEFECTIVE" : "NORMAL");
                    ImGui::Text("%s: %.4f",
                                processor.config().kind == "rf" ? "Tree vote fraction"
                                                                : "SVM margin",
                                p.result.score);
                    ImGui::TextDisabled("Validation threshold: %.4f", processor.config().threshold);
                    ImGui::TextWrapped("Score is uncalibrated. Candidate regions are suspected "
                                       "areas, separate from classification.");
                    if (p.frame >= 0) {
                        ImGui::Text("Video frame %d / %d", p.frame, p.totalFrames);
                        if (ImGui::Button(playing ? "Pause" : "Play")) {
                            playing = !playing;
                            videoNext = glfwGetTime();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Step frame") && !pending) {
                            playing = false;
                            request(2);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Rewind") && !pending) {
                            playing = false;
                            request(3, "", 0);
                        }
                        ImGui::TextDisabled("Source rate %.1f fps; analyzed playback", p.fps);
                    }
                }
                ImGui::Separator();
                const char *views[] = {"Original", "Processed (CLAHE)", "Predicted overlay"};
                ImGui::Combo("View", &mode, views, 3);
                ImGui::Checkbox("Candidate bounding boxes", &showBoxes);
                ImGui::BeginDisabled(!groundTruthAvailable);
                ImGui::Checkbox("Ground truth (evaluation only)", &showGT);
                ImGui::EndDisabled();
                ImGui::SliderFloat("Zoom", &zoom, 0.2f, 6.f, "%.2fx");
                ImGui::SliderFloat("Rotation", &angle, -180, 180, "%.0f deg");
                if (ImGui::Button("Reset / fit to view")) {
                    zoom = 1;
                    angle = 0;
                    pan = {};
                    roi.reset();
                    selected = -1;
                }
                ImGui::TextDisabled("Wheel: zoom  |  right/middle drag: pan");
                ImGui::TextDisabled("Left click: pick a candidate region");
                ImGui::Checkbox("Draw rectangular ROI (left drag)", &roiMode);
                ImGui::Checkbox("Liang-Barsky clip at ROI", &clipROI);
                if (ImGui::Button("Clear ROI"))
                    roi.reset();
                if (roi)
                    ImGui::Text("ROI pixels: (%.0f,%.0f) to (%.0f,%.0f)", roi->x0, roi->y0, roi->x1,
                                roi->y1);
                if (displayed) {
                    ImGui::Text("Candidate regions: %d", int(displayed->result.regions.size()));
                    if (selected >= 0 && selected < int(displayed->result.regions.size())) {
                        auto &r = displayed->result.regions[selected];
                        ImGui::TextColored({1, 0.86f, 0.3f, 1}, "Selected suspected region %d",
                                           selected + 1);
                        ImGui::Text("Area: %d pixels", r.pixelArea);
                        ImGui::Text("Box: x=%d y=%d w=%d h=%d", r.box.x, r.box.y, r.box.width,
                                    r.box.height);
                    }
                    ImGui::TextDisabled("Features %.2f ms / prediction %.3f ms",
                                        displayed->result.featureMs, displayed->result.predictMs);
                    ImGui::TextDisabled("Localization + processed view %.2f ms",
                                        displayed->result.localizeMs);
                }
                ImGui::TextDisabled("Rendering %.2f ms  |  event rate %.1f fps", lastRenderMs,
                                    io.Framerate);
                if (smoke && smokeFrames >= 81)
                    ImGui::SetNextItemOpen(true, ImGuiCond_Always);
                if (ImGui::CollapsingHeader("About / team")) {
                    ImGui::TextWrapped(
                        "AI-Assisted Surface Defect Detection and Interactive Visualization");
                    ImGui::TextWrapped("IIITDM Kancheepuram\nGyan Chandra - CS23I1053\nVaibhav - "
                                       "CS23I1040\nVishal Singh - CS23I1039");
                    ImGui::TextWrapped("C++17 / OpenCV / OpenGL 3.3 / GLFW / GLEW / Dear "
                                       "ImGui\nKolektorSDD2: CC BY-NC-SA 4.0");
                    ImGui::TextWrapped("A dataset-stills video is an image-sequence demonstration, "
                                       "not independent video evaluation.");
                }
                ImGui::End();
                if (displayed) {
                    gfx::Point mouse{io.MousePos.x, io.MousePos.y};
                    bool inCanvas = viewport.contains(mouse) && !io.WantCaptureMouse;
                    if (inCanvas) {
                        if (io.MouseWheel != 0) {
                            auto imagePoint = transform.inverse().apply(mouse);
                            zoom =
                                std::clamp(zoom * float(std::pow(1.15, io.MouseWheel)), 0.2f, 6.f);
                            auto next = gfx::imageTransform(displayed->image.cols,
                                                            displayed->image.rows, viewport, zoom,
                                                            angle * 3.141592653589793 / 180, pan);
                            auto moved = next.apply(imagePoint);
                            pan.x += mouse.x - moved.x;
                            pan.y += mouse.y - moved.y;
                        }
                        if (ImGui::IsMouseDragging(1) || ImGui::IsMouseDragging(2)) {
                            pan.x += io.MouseDelta.x;
                            pan.y += io.MouseDelta.y;
                        }
                        if (ImGui::IsMouseClicked(0)) {
                            if (roiMode || io.KeyShift) {
                                dragROI = true;
                                roiStart = transform.inverse().apply(mouse);
                            } else
                                selected = pick(*displayed, transform, mouse,
                                                clipROI ? roi : std::optional<gfx::Rect>{});
                        }
                    }
                    if (dragROI) {
                        auto p = transform.inverse().apply(mouse);
                        p.x = std::clamp(p.x, 0.0, double(displayed->image.cols));
                        p.y = std::clamp(p.y, 0.0, double(displayed->image.rows));
                        roi = gfx::Rect{std::clamp(std::min(roiStart.x, p.x), 0.0,
                                                   double(displayed->image.cols)),
                                        std::clamp(std::min(roiStart.y, p.y), 0.0,
                                                   double(displayed->image.rows)),
                                        std::clamp(std::max(roiStart.x, p.x), 0.0,
                                                   double(displayed->image.cols)),
                                        std::clamp(std::max(roiStart.y, p.y), 0.0,
                                                   double(displayed->image.rows))};
                        if (!ImGui::IsMouseDown(0))
                            dragROI = false;
                    }
                    transform =
                        gfx::imageTransform(displayed->image.cols, displayed->image.rows, viewport,
                                            zoom, angle * 3.141592653589793 / 180, pan);
                }
                if (playing && displayed && displayed->frame >= 0 && !pending &&
                    glfwGetTime() >= videoNext) {
                    if (displayed->frame + 1 >= displayed->totalFrames)
                        playing = false;
                    else
                        request(2);
                }
                std::string capture;
                if (smoke && displayed && !pending && !smokeVideo) {
                    smokeFrames++;
                    if (smokeFrames == 1) {
                        mode = 0;
                        showGT = false;
                    }
                    if (smokeFrames == 20) {
                        capture = "01_original.png";
                        shots++;
                    }
                    if (smokeFrames == 21)
                        mode = 1;
                    if (smokeFrames == 40) {
                        capture = "02_processed.png";
                        shots++;
                    }
                    if (smokeFrames == 41) {
                        mode = 2;
                        angle = 32;
                        zoom = 1.25;
                        pan = {40, -15};
                        showGT = true;
                        auto &regions = displayed->result.regions;
                        if (!regions.empty()) {
                            auto largest =
                                std::max_element(regions.begin(), regions.end(),
                                                 [](const ai::Region &a, const ai::Region &b) {
                                                     return a.pixelArea < b.pixelArea;
                                                 });
                            auto &r = *largest;
                            roi = gfx::Rect{double(r.box.x), double(std::max(0, r.box.y - 10)),
                                            double(displayed->image.cols - 2),
                                            double(std::min(displayed->image.rows,
                                                            r.box.y + r.box.height + 80))};
                            auto local = gfx::Point{double(r.contour[0].x), double(r.contour[0].y)};
                            auto m = gfx::imageTransform(displayed->image.cols,
                                                         displayed->image.rows, viewport, zoom,
                                                         angle * 3.141592653589793 / 180, pan);
                            selected = pick(*displayed, m, m.apply(local), roi);
                            if (selected < 0)
                                throw std::runtime_error("Transformed candidate picking failed");
                            roi->x0 += r.box.width * 0.35;
                        }
                    }
                    if (smokeFrames == 60) {
                        capture = "03_rotation_roi_overlay.png";
                        shots++;
                    }
                    if (smokeFrames == 61) {
                        glfwSetWindowSize(window, 1100, 760);
                        angle = -18;
                        zoom = 0.85;
                        pan = {-30, 20};
                        roi.reset();
                        showGT = false;
                    }
                    if (smokeFrames == 80) {
                        capture = "04_resized_overlay.png";
                        shots++;
                    }
                    if (smokeFrames == 81) {
                        glfwSetWindowSize(window, 1280, 950);
                        angle = 0;
                        zoom = 1;
                        pan = {};
                    }
                    if (smokeFrames == 83) {
                        capture = "06_about_team.png";
                        shots++;
                    }
                    if (smokeFrames == 84) {
                        auto video = output / "demo_sequence.avi";
                        if (!ai::fs::exists(video))
                            throw std::runtime_error(
                                "Smoke test requires demo-video command first");
                        smokeVideo = true;
                        request(1, video.string());
                    }
                }
                if (smoke && smokeVideo && displayed && displayed->frame >= 0 && !pending) {
                    if (smokeVideoFrames < 4)
                        request(2);
                    else if (smokeVideoFrames == 4) {
                        mode = 2;
                        roi.reset();
                        showGT = false;
                        capture = "05_video_frame.png";
                        shots++;
                        glfwSetWindowShouldClose(window, true);
                    }
                }
                // Evaluation-only geometry is decoded once per source, outside render timing.
                // No ground truth enters the prediction worker or feature construction.
                if (showGT && groundTruthAvailable && displayed &&
                    groundTruthToken != displayed->token) {
                    auto gt = cv::imread(groundTruthPath.string(), cv::IMREAD_GRAYSCALE);
                    if (gt.empty() || gt.size() != displayed->image.size()) {
                        groundTruthAvailable = false;
                        showGT = false;
                        error = "Evaluation mask is unavailable or has incorrect dimensions";
                    } else {
                        cv::findContours(gt, groundTruthContours, cv::RETR_EXTERNAL,
                                         cv::CHAIN_APPROX_SIMPLE);
                        groundTruthToken = displayed->token;
                    }
                }
                auto renderStart = std::chrono::steady_clock::now();
                glViewport(0, 0, fw, fh);
                glDisable(GL_DEPTH_TEST);
                glClearColor(0.035f, 0.045f, 0.06f, 1);
                glClear(GL_COLOR_BUFFER_BIT);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                if (displayed) {
                    auto &p = *displayed;
                    transform = gfx::imageTransform(p.image.cols, p.image.rows, viewport, zoom,
                                                    angle * 3.141592653589793 / 180, pan);
                    renderer.setTransform(transform, float(ww), float(wh));
                    if (uploadedMode != mode) {
                        renderer.upload(mode == 1 ? p.result.processed : p.image);
                        uploadedMode = mode;
                    }
                    float sx = float(fw) / ww;
                    glEnable(GL_SCISSOR_TEST);
                    glScissor(int(sidebar * sx), 0, int((ww - sidebar) * sx), fh);
                    renderer.image(p.image.cols, p.image.rows);
                    auto clipping = clipROI ? roi : std::optional<gfx::Rect>{};
                    if (mode == 2)
                        for (int i = 0; i < int(p.result.regions.size()); i++) {
                            auto &r = p.result.regions[i];
                            renderer.lines(segments(r.contour, clipping),
                                           i == selected ? ImVec4{1, 0.95f, 0.25f, 1}
                                                         : ImVec4{1, 0.36f, 0.13f, 1},
                                           i == selected ? 2 : 1);
                            if (showBoxes)
                                renderer.lines(segments(boxPolygon(r.box), clipping),
                                               i == selected ? ImVec4{1, 0.95f, 0.25f, 1}
                                                             : ImVec4{1, 0.65f, 0.20f, 1},
                                               1);
                        }
                    if (showGT && groundTruthToken == p.token)
                        for (auto &c : groundTruthContours)
                            renderer.lines(segments(c, {}), {0.82f, 0.35f, 1, 1}, 2);
                    if (roi) {
                        std::vector<cv::Point> polygon{{int(roi->x0), int(roi->y0)},
                                                       {int(roi->x1), int(roi->y0)},
                                                       {int(roi->x1), int(roi->y1)},
                                                       {int(roi->x0), int(roi->y1)}};
                        renderer.lines(segments(polygon, {}), {0.25f, 0.95f, 0.63f, 1}, 2);
                    }
                    glDisable(GL_SCISSOR_TEST);
                }
                auto *foreground = ImGui::GetForegroundDrawList();
                foreground->AddText({sidebar + 24, 24}, IM_COL32(190, 206, 219, 255),
                                    mode == 0   ? "ORIGINAL IMAGE"
                                    : mode == 1 ? "PROCESSED / CONTRAST ENHANCED"
                                                : "PREDICTED CANDIDATES / ORANGE");
                if (showGT)
                    foreground->AddText({sidebar + 24, 44}, IM_COL32(215, 120, 255, 255),
                                        "GROUND TRUTH / PURPLE / EVALUATION ONLY");
                if (roi)
                    foreground->AddText({sidebar + 24, float(wh) - 32}, IM_COL32(80, 240, 160, 255),
                                        "GREEN ROI / OVERLAY SEGMENTS CLIPPED WITH LIANG-BARSKY");
                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                if (smoke)
                    glFinish();
                lastRenderMs = ai::milliseconds(renderStart);
                if (displayed)
                    renderTimes.push_back(lastRenderMs);
                if (renderTimes.size() > 10000)
                    renderTimes.erase(renderTimes.begin(), renderTimes.begin() + 5000);
                if (!capture.empty())
                    screenshot(output / "screenshots" / capture, fw, fh);
                if (capture == "01_original.png")
                    textureSampleError = verifyTexture(*displayed, transform, ww, wh, fw, fh);
                GLenum e = glGetError();
                if (e != GL_NO_ERROR)
                    throw std::runtime_error("OpenGL rendering error " + std::to_string(e));
                glfwSwapBuffers(window);
                if (smoke && ai::milliseconds(start) > 90000)
                    throw std::runtime_error("Graphics smoke test timed out");
            }
            if (smoke) {
                if (shots != 6 || smokeVideoFrames != 4 || textureSampleError < 0)
                    throw std::runtime_error("Incomplete graphics smoke test");
                std::sort(renderTimes.begin(), renderTimes.end());
                double sum = std::accumulate(renderTimes.begin(), renderTimes.end(), 0.0);
                cv::FileStorage f((output / "graphics_smoke.json").string(),
                                  cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
                f << "opengl_version" << glVersion << "renderer" << glRenderer
                  << "genuine_framebuffer_screenshots" << shots << "rendered_frames"
                  << int(renderTimes.size()) << "render_mean_ms" << sum / renderTimes.size()
                  << "render_p95_ms" << renderTimes[size_t(.95 * (renderTimes.size() - 1))]
                  << "render_mean_throughput_fps" << 1000 * renderTimes.size() / sum
                  << "render_timing_includes_gpu_finish" << 1 << "analysis_mean_ms"
                  << analysisTotal / std::max(1, analyzed) << "video_frames_rendered"
                  << smokeVideoFrames << "transform_picking_resize_roi_exercised" << 1
                  << "framebuffer_source_pixel_samples" << 9
                  << "framebuffer_source_max_channel_error" << textureSampleError;
                std::cout << "Graphics smoke passed: " << shots
                          << " framebuffer captures, rotation/zoom/pan/ROI/picking/resize, 4 video "
                             "frames; render "
                          << sum / renderTimes.size() << " ms\n"
                          << std::flush;
            }
        } catch (const std::exception &e) {
            std::cerr << "VIEWER ERROR: " << e.what() << '\n';
            code = 1;
        }
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return code;
}
} // namespace app
