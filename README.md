# AI-Assisted Surface Defect Detection and Interactive Visualization

**Theme 6: Computer Vision-Based Image and Video Analysis with Graphics Visualization**

Interactive Computer Graphics course project — **IIITDM Kancheepuram**.

| Team member | Roll number |
|---|---|
| Gyan Chandra | CS23I1053 |
| Vaibhav | CS23I1040 |
| Vishal Singh | CS23I1039 |

This README describes the current implementation, inspected on **18 October 2026**. Experiment figures are **saved results** from the existing source, model, manifests, JSON/CSV outputs and logs. Training, full evaluation, downloads and GUI smoke tests were not rerun for this documentation update. The lightweight checks rerun during this audit are listed in section 12.

## 1. Project overview

The application helps inspect industrial surface images for visible defects. Inputs are an image, a local video file, or images browsed from KolektorSDD2. Outputs are:

- An image-level **normal/defective** classification, an uncalibrated model score and the validation-selected threshold.
- A separate binary **candidate-region mask**, suspected-region contours, axis-aligned boxes and pixel areas.
- An OpenGL canvas with original, processed and candidate-overlay views, transformations, region picking and a rectangular ROI.
- Headless training/evaluation commands and saved experiment outputs.

There are two analysis branches. The classifier converts image pixels into 110 intensity, gradient and morphological-response features, standardizes them with fit-set statistics and applies the saved SVM. The candidate method uses morphological responses, gradient support and connected components. The viewer displays both results, but **classification does not generate the candidate mask**. A normal classification can have candidate regions, and a defective classification can have an empty or incomplete mask.

| Implementation area | Main modules |
|---|---|
| Dataset, DIP/CV, features and ML | [pipeline.hpp](src/pipeline.hpp), [pipeline.cpp](src/pipeline.cpp), [evaluation.cpp](src/evaluation.cpp) |
| OpenGL rendering and interaction | [geometry.hpp](src/geometry.hpp), [geometry.cpp](src/geometry.cpp), [viewer.cpp](src/viewer.cpp) |
| Integration, video and verification | [integration.hpp](src/integration.hpp), [integration.cpp](src/integration.cpp), [main.cpp](src/main.cpp), [core_tests.cpp](tests/core_tests.cpp) |

These are module responsibilities; no individual contribution history is assigned to team members.

## 2. Implementation status

**Executed** means supported by saved execution evidence or the listed lightweight checks. **Code present** means the path exists but the relevant manual interaction/compatibility has not been verified. **Limited** identifies a working feature with a material scope or quality limitation.

| Feature | Status | Evidence and boundary |
|---|---|---|
| Dataset pairing, labels, validation and duplicate audit | Executed | [Manifest](outputs/manifest.csv), [audit](outputs/dataset_audit.txt), and current CSV/split/file-presence cross-checks agree. Full decoding was not repeated for this audit. |
| Weighted RF/SVM comparison, baseline, ablation and threshold selection | Executed | Eight fitted candidates plus baseline in [model_comparison.csv](outputs/model_comparison.csv); only the selected SVM is retained as a model. |
| Saved-model image-only inference and reload consistency | Executed | [Training summary](outputs/training_summary.json), [runtime checks](outputs/runtime_checks.json), XML/YAML and a fresh read-only prediction. Viewer launch does not train. |
| Candidate masks, contours, boxes and pixel areas | Executed, limited quality | Separate image-based method; [localization results](outputs/test_localization.csv). Mean defective-image Dice is 0.295894. |
| OpenGL canvas, overlays, transforms, clipping and picking | Executed | [Graphics smoke record](outputs/graphics_smoke.json), six saved framebuffer captures and geometry tests. |
| Physical mouse gestures, ROI drawing and GUI sliders | Code present; math/rendering exercised | Smoke mode sets transform/ROI state and calls picking programmatically. Physical gestures were not manually verified. |
| Native dialogs, dropdown browsing and drag/drop | Code present; manual use unverified | Handlers exist and widgets appear in screenshots. Saved loading checks use explicit file paths. |
| Video decoding, successive-frame analysis and seeking | Executed | Six decoded MJPEG frames and a seek checked against a separate decoder; four frames exercised by graphics smoke mode. |
| Play/pause, step and rewind buttons | Code present; manual button use unverified | Buttons use the verified worker request paths. There is no arbitrary-frame seek slider. |
| Rectangular ROI | Executed for visualization/picking | ROI clips overlays and restricts picking; it does not crop input or recompute classification/localization. |
| Independent natural-video accuracy/temporal analysis | Not evaluated | The demonstration is made from dataset stills. There is no video accuracy or tracking experiment. |
| Portable Windows trial | Executed on development machine | [Package verification](outputs/portable_package_verification.json): isolated-path checks, 144 DLLs, 29 real image/mask pairs. Other machines remain unverified. |
| Non-Windows/MSVC compatibility | Unverified | Setup/build scripts target Windows MinGW64. Non-Windows native-dialog code returns an empty path. A CMake MSVC branch is not a tested toolchain. |

## 3. Algorithm inventory

DIP changes pixel values; CV derives measurements or regions from those pixels. Several operations serve both roles. **YAML** parameters come from [pipeline.yml](outputs/model/pipeline.yml); **code** parameters are fixed version-1 constants.

### DIP and CV

| Area | Exact method | Purpose | Parameters / configuration | Input → output | Source / function |
|---|---|---|---|---|---|
| DIP | BGR/BGRA-to-grayscale conversion; gray pass-through | Obtain one intensity plane | 8-bit input; channel handling in code; app loaders normally provide BGR | Native image → gray | [pipeline.cpp](src/pipeline.cpp), `preprocess` |
| DIP | Area-interpolation resizing, `INTER_AREA` | Use a consistent analysis grid | YAML `width=224`, `height=608` | Native gray → 224 × 608 image | `preprocess` |
| DIP | Gaussian smoothing | Reduce small fluctuations before measurements | YAML `denoise_kernel=3`, `denoise_sigma=0.7` | Resized gray → denoised gray | `preprocess` |
| DIP | CLAHE | Enhance local contrast for gradients/display | YAML `clahe_clip=2`; tile grid 8 × 16 | Denoised gray → enhanced gray | `preprocess` |
| DIP / CV | Sobel X/Y and Euclidean gradient magnitude | Measure edge strength; support candidate pixels | Code: 3 × 3 Sobel, `CV_32F`, magnitude divided by 4 | Enhanced gray → gradient magnitude | `preprocess` |
| DIP / CV | Gaussian background and absolute residual | Measure departures from smooth intensity | YAML `background_sigma=5`; automatic kernel size | Denoised gray → absolute background difference | `preprocess` |
| DIP / CV | Morphological black-hat and white top-hat | Detect darker/brighter local structures | YAML `morphology_kernel=17`; elliptical 17 × 17 element | Denoised gray → dark/bright response maps | `features`, `localize` |
| CV | Global distribution statistics | Summarize response distributions | Code: mean, population std, four quantiles, three threshold fractions; section 5 | Plane → 9 values | `stats` |
| CV | Fixed-cell intensity statistics | Retain coarse spatial variation | Code: 3 columns × 4 rows; mean/std per cell | Enhanced image → 24 values | `features` |
| CV | Ranked non-overlapping patch statistics | Capture strong local responses without target-dependent crops | YAML `patch_size=24`; code: offset 8, complete patches, ranks 1/2/5/10 | Four maps → 32 values | `features` |
| CV | Fixed binary threshold | Form candidate foreground | Maximum of black-/top-hat; YAML `local_threshold=30`; `THRESH_BINARY`, values 0/255; strict `response > 30` | Response → binary mask | `localize` |
| DIP / CV | Gradient threshold, dilation and binary intersection | Require nearby edge support | Cast gradient to 8-bit; YAML `gradient_support=4`; code: >4, 3 × 3 square dilation, bitwise AND | Gradient + candidates → supported mask | `localize` |
| DIP / CV | Binary morphological closing | Join short gaps | Code: 3 × 3 square element | Supported mask → closed mask | `localize` |
| CV | Border suppression | Remove boundary responses | YAML `margin=8` pixels on each side | Closed mask → interior-only mask | `localize` |
| CV | Connected components with statistics and area filtering | Remove small connected regions | OpenCV default 8-connectivity; YAML `min_area=24` working-grid pixels; retain area ≥ 24 | Binary mask → labels/stats → filtered mask | `localize`; computed centroids are unused |
| DIP / CV | Nearest-neighbour mask restoration; linear display resizing | Restore native image coordinates | Code: `INTER_NEAREST` for mask; `INTER_LINEAR` for display | Working output → native-size mask/display | `localize`, `Model::analyze` |
| CV | External contour tracing and chain simplification | Represent region boundaries | `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`; holes are not separate contours | Native mask → contours | `localize` |
| CV | Axis-aligned bounding rectangle and raster area count | Supply region details | `boundingRect`; fill contour, intersect with mask, `countNonZero` | Contour + mask → box/pixel area | `localize`; `Region` in [pipeline.hpp](src/pipeline.hpp) |
| CV / data handling | Decoded-source SHA-256 grouping | Keep identical sources together | Windows BCrypt hashes dimensions/type + BGR bytes; non-Windows fallback is FNV-style, not SHA-256 | Source image → grouping key | [evaluation.cpp](src/evaluation.cpp), `pixelHash`, `manifest`, `trainEvaluate` |

Here, “texture features” means the residual, morphological-response and patch statistics listed above. The gradient method is Sobel magnitude; candidate segmentation uses fixed thresholds and morphological cleanup.

### AI/ML and selection

| Area | Exact method | Purpose | Parameters / configuration | Input → output | Source / function |
|---|---|---|---|---|---|
| AI/ML | Fit-set column-wise z-score standardization | Compare differently scaled feature columns | Save fit mean and `max(std, 1e-5)` in YAML; `CV_32F` | Features → standardized features | [pipeline.cpp](src/pipeline.cpp), `Model::fitScale`, `normalize` |
| AI/ML | Inverse-frequency positive-class weight | Address imbalance during fitting | `min(10, negatives/positives)` from fit labels; normal weight 1; saved positive weight 8.46700478 | Fit labels → RF priors / SVM weights | `Model::train` |
| AI/ML | Random Forest, OpenCV `ml::RTrees` | Compare a nonlinear tree ensemble | 200 trees; depth 10/16; min samples 3; max categories 2; active-variable count 0 (automatic); seed 20261013 | Standardized features → class votes | `Model::train`, `scores`; importance calculation is enabled but no importance report is exported |
| AI/ML | RBF SVM, `ml::SVM::C_SVC` | Produce image-level decision scores | C=1/10, gamma=1/110; 3000 iterations / epsilon `1e-5` | Standardized features → raw margin | `Model::train`, `scores` |
| AI/ML | Training-derived margin orientation | Make larger scores correspond to defects | Compare mean fit-class raw scores; saved YAML `orientation=-1` | Raw margin → oriented margin | `Model::train`, `scores` |
| AI/ML | Majority-normal baseline | Show the effect of imbalance | Constant score 0; threshold 1 | Label list → all-normal predictions | [evaluation.cpp](src/evaluation.cpp), `trainEvaluate`, `finalTest` |
| AI/ML | Validation threshold search and F2 selection | Choose feature variant, model and operating threshold | Distinct validation scores; defect if `score >= threshold`; maximize `F2 + 1e-6 × F1` | Validation scores/labels → selected pipeline | `chooseThreshold`, `trainEvaluate` |
| CV / selection | Localization validation grid search | Balance overlap and normal-image foreground | Thresholds 12/20/30/45 × min areas 8/24; maximize mean defective Dice − 2 × normal foreground fraction | Validation image/mask pairs → settings | `localization`, `trainEvaluate` |

### ICG

| Area | Exact method | Purpose | Parameters / configuration | Input → output | Source / function |
|---|---|---|---|---|---|
| ICG | RGB texture mapping on two triangles | Draw the custom image canvas | `GL_RGB8`, bilinear sampling, clamp-to-edge, unpack alignment 1; GLSL 330 core | BGR/gray pixels → textured quad | [viewer.cpp](src/viewer.cpp), `Renderer::upload`, `image` |
| ICG | Homogeneous 2D affine transformation | Fit, scale, rotate and pan image/overlays together | CPU `Mat3`; column-major shader upload; fit uses rotated extent | Image → window → normalized-device coordinates | [geometry.cpp](src/geometry.cpp), `imageTransform`; `Renderer::uniforms` |
| ICG / CV | Affine inverse and contour inclusion test | Pick the region under the pointer | Inverse mapping; optional ROI containment; reverse region order; `pointPolygonTest >= 0` | Screen point → image point → region index | [viewer.cpp](src/viewer.cpp), `pick` |
| ICG | Liang–Barsky parametric line clipping | Clip candidate segments at ROI boundary | Four rectangle inequalities; parallel tolerance `1e-12`; before transformation | Segment + ROI → shortened segment/rejection | [geometry.cpp](src/geometry.cpp), `liangBarsky`; `segments` in viewer |
| ICG | Line primitives and viewport scissor | Draw contours, boxes and ROI inside canvas | `GL_LINES`; scissor scaled to framebuffer width | Segments → overlays | `Renderer::lines`, `viewer` |
| ICG / test helper | Ray-crossing point-in-polygon | Test transformed polygon inclusion | Ray crossings toggle inside/outside | Point + polygon → boolean | `gfx::pointInPolygon` in [geometry.cpp](src/geometry.cpp); used by tests, **not** actual viewer picking |

## 4. Dataset and split handling

### Source, files and verified counts

The dataset is [KolektorSDD2, distributed by ViCoS Lab](https://www.vicos.si/resources/kolektorsdd2/). The recorded direct download is [KolektorSDD2.zip](https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip).

~~~text
data/
  KolektorSDD2.zip
  source.json
  KolektorSDD2/
    train/<id>.png, <id>_GT.png
    test/<id>.png, <id>_GT.png
    split_weakly_*.pyb
~~~

[Source metadata](data/source.json) records 853,126,555 archive bytes, download time `2026-10-13T08:26:21.315492+00:00`, successful ZIP CRC verification and SHA-256:

~~~text
edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8
~~~

These are saved integrity results, not a new archive validation during this documentation task.

The following counts were recomputed from the existing [manifest](outputs/manifest.csv) and [split CSV](outputs/development_split.csv). They agree with saved results; all referenced image/mask files are present.

| Set | Normal (0) | Defective (1) | Total |
|---|---:|---:|---:|
| Official training | 2,085 | 246 | 2,331 |
| Fit subset | 1,668 | 197 | 1,865 |
| Validation subset | 417 | 49 | 466 |
| Official test | 894 | 110 | 1,004 |
| All canonical source images | 2,979 | 356 | 3,335 |

Manifest dimensions range from **184–241 pixels wide** and **597–665 pixels high**. These ranges come from saved decoded-image metadata; the viewer preserves native image dimensions.

`manifest()` accepts numeric PNG names in `train`/`test`, pairs each with `<id>_GT.png`, decodes both, checks equal dimensions and derives `label = countNonZero(mask) > 0`. It rejects missing pairs, decode failures, orphan masks, conflicting duplicate-group labels and counts different from the canonical dataset. It is therefore not a generic arbitrary-dataset validator.

Auxiliary entries are preserved and excluded from labelled samples:

- `train/10301 (copy).png` and `train/10301_GT (copy).png`.
- `split_weakly_0.pyb`, `split_weakly_16.pyb`, `split_weakly_53.pyb`, `split_weakly_126.pyb`, `split_weakly_246.pyb`.

The saved audit identifies the extra image as a duplicate of `train/10301`. This documentation audit additionally compared file bytes: **both copy files are identical to their corresponding canonical image/mask**. The audit message “not identical to canonical pixels” for the mask copy compares it against source-image hashes, not its paired canonical mask; it does not establish a changed annotation. No `.pyb` is executed or deserialized.

### Split procedure and leakage checks

`trainEvaluate()` retains the official test split, groups decoded-identical sources, excludes training sources whose hashes occur in test, shuffles groups separately by class using `std::mt19937(20261013)` and assigns approximately 20% of each class's groups to validation. Complete source groups stay together. Patch statistics are aggregated inside an already assigned source image, rather than split/trained as separate samples.

The current manifest has **no duplicate canonical hash groups**. Fit/validation hashes are disjoint, neither subset overlaps test, and the training summary records zero exclusions for test overlap. The split CSV records the exact assignments. With duplicates, rounding is by group count rather than exact image percentage.

Masks supply supervised labels, validation localization targets, held-out evaluation and optional labelled ground-truth display. `features()`, `localize()` and `Model::analyze()` receive image pixels and configuration, without filenames, masks or labels. Scaling and imbalance weights use only fit samples. New-image inference needs the image plus saved classifier/configuration.

The manifest reads test masks for pairing/label validation and evaluation; they do not enter feature construction, fitting or validation model selection. The viewer can choose the first reported true-positive image from the saved CSV as a startup demonstration. This sample choice is separate from inference and is not an unbiased new accuracy test. No target-feature leakage was found in the inspected execution paths.

## 5. Features, classifiers and inference

### Exact feature vector

One image produces a `1 × 110`, `CV_32F` row. Indices are zero-based.

| Columns | Count | Contents |
|---|---:|---|
| 0–8 | 9 | Denoised-intensity distribution |
| 9–17 | 9 | CLAHE-intensity distribution |
| 18–26 | 9 | Sobel-magnitude/4 distribution |
| 27–50 | 24 | Enhanced-image mean/std in 3 × 4 fixed cells |
| 51–77 | 27 | Nine values each for absolute Gaussian residual, black-hat and top-hat |
| 78–109 | 32 | Ranked patch means/maxima from residual, gradient, black-hat and top-hat |

Each distribution descriptor has mean, population std, quantiles 0.5/0.9/0.99/0.999 and fractions strictly above 8/16/32. Quantiles select sorted element `floor(q × (n−1))` without interpolation. Mean/std/quantiles are divided by 255; threshold fractions use pixel count. Spatial means/stds and ranked patch means/maxima are also divided by 255. Patch output within each map is mean/max pairs for ranks 1, 2, 5 and 10. This is fixed unit scaling. Gradient features can exceed 1, so “bounded” means **fixed feature count**, not clipping every value to [0,1].

Patches are complete, non-overlapping 24 × 24 blocks starting at (8,8), with an 8-pixel far-border constraint. At 224 × 608 there are **192 patches per response map**. Means/maxima are independently sorted descending; ranks 1, 2, 5 and 10 are retained. Edge strips remain uncovered where complete patches do not fit. Spatial cells use integer rectangles: at default width they are 74 pixels wide and two columns are omitted from these cell summaries. Global statistics still cover the full working image.

The ablation preserves 110 columns but zeroes columns 51–109. It removes 59 residual/texture values while retaining global intensity/gradient and spatial summaries. Each candidate fits its own scaler/model on the same fit subset.

### Fitting and selection

~~~text
z = (feature − fit_mean) / max(fit_population_std, 1e-5)
positive_weight = min(10, fit_normal_count / fit_defective_count)
~~~

Fit counts produce weights approximately `[1, 8.46700478]` for normal/defective. RF uses priors and SVM uses class weights. Two feature variants × four candidates give eight fits: RF depth 10/16 and RBF SVM C=1/10. RF has 200 trees and min samples 3; SVM gamma is 1/110 with 3000-iteration/`1e-5` limits. The seed is reset to 20261013 before each fit.

Fixed preprocessing constants are not independently searched. Feature inclusion, classifier parameters, score threshold and separate localization parameters are selected on validation. Distinct validation scores form the threshold candidates; defect means `score >= threshold`. Threshold/model selection maximizes `F2 + 1e-6 × F1`. F2 is `5PR/(4P+R)` and gives recall more weight than F1. Accuracy is not the selection objective.

The winner is **full-feature SVM, C=1**, trained on 1,865 images. It is **not refitted on validation**, preserving the selected score/threshold relationship. The training path saves this pipeline before final testing. Other candidates have validation records but no retained RF XML or individual RF final-test reports.

### Saved model and scores

| File | Actual contents |
|---|---|
| [classifier.xml](outputs/model/classifier.xml) | C-SVC, RBF, C=1, gamma=0.0090909090909090905, 110 variables, labels 0/1, weights 1/8.46700478, **756 support vectors** |
| [pipeline.yml](outputs/model/pipeline.yml) | Version 1; preprocessing/localization fields; full-feature flag; `1 × 110` fit means/scales; classifier kind and decision settings |

The active score is `−raw_margin` because saved `orientation=-1`. Its threshold is **−0.081807687878608704**. This is an uncalibrated margin, not a probability. RF scores are positive-class tree-vote fractions and are also treated as uncalibrated.

Version-1 source defines constants not stored as adjustable YAML fields, including Sobel size, descriptor quantiles/levels, cells and patch ranks. Use the saved configuration with its matching feature implementation. `Model::load()` validates configuration/scaler basics; `normalize()` rejects feature-dimension mismatches.

`Model::analyze()` extracts features, standardizes/predicts, independently computes native-size candidate masks/regions and prepares the CLAHE display. Thus CLI `predict` also performs localization/display preparation, though it only prints a summary. Neither `predict` nor `view` starts training.

The [training summary](outputs/training_summary.json) records **maximum validation score difference 0** after save/reload. The training code also reconstructs features from up to 12 validation images with the loaded configuration and requires maximum difference ≤ `1e-6`; completion of final outputs supports that check having passed. Runtime evidence separately confirms inference from an image copied without its annotation.


## 6. Candidate-region localization

Localization is a separate image-processing method in `ai::localize()` ([pipeline.cpp](src/pipeline.cpp)). It does not turn the SVM score into a segmentation mask and does not use the ground-truth mask at inference.

1. Convert and resize the image to the configured working grid; denoise it as in section 3.
2. Compute dark black-hat and bright top-hat responses with the `17 × 17` elliptical kernel. Take their pixelwise maximum so that both dark and bright local anomalies can be candidates.
3. Apply a strict response threshold `> 30` using the saved configuration.
4. Convert the Sobel magnitude divided by four to 8-bit, threshold it `> 4`, dilate with a `3 × 3` square and AND it with the candidate mask. This requires nearby gradient support.
5. Close the result with a `3 × 3` square and clear an eight-pixel border.
6. Use eight-connected components and retain components with at least `24` working-grid pixels.
7. Resize the binary mask to the input image dimensions with nearest-neighbour interpolation.

These are the **saved model's active settings**. The C++ defaults before loading a configuration use response threshold 18 and minimum area 12. Validation searched response thresholds `{12, 20, 30, 45}` and minimum areas `{8, 24}`; it selected `30/24` by maximizing:

~~~text
mean Dice on defective validation images
− 2 × mean foreground-area fraction on normal validation images
~~~

All eight measured settings are in [localization_validation.csv](outputs/localization_validation.csv). This is validation-based selection of a hand-designed candidate method; it is not a learned segmentation network.

When regions are requested, `findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` extracts external boundaries from the native-size mask. Each region stores its contour, `boundingRect` and a pixel count. The count fills that external contour into a temporary mask, intersects it with the predicted mask and counts nonzero pixels. It is **not** `contourArea()`, square millimetres or a calibrated defect size. Holes have no separate contour representation. Component centroids are computed by OpenCV but not used.

The viewer marks candidate contours/boxes in orange and the selected region in yellow. Ground truth, when explicitly enabled for a paired dataset image, is purple and labelled for evaluation only. Selection reports a suspected region and its area in native-image pixels. Classification and localization can disagree because their algorithms are independent.

The rectangular ROI affects visible candidate segments and picking only. It does not crop classifier input, rerun localization, filter the reported mask or change any saved metric.

### Localization evaluation

[evaluation.cpp](src/evaluation.cpp) compares the native-size predicted mask with the paired ground truth, binarized with `GT > 0`:

- **Mean defective IoU and Dice:** calculate each defective image's metric and average over defective images; an empty prediction against a nonempty ground truth scores zero.
- **Defective pixel precision/recall:** aggregate pixel TP/FP/FN over defective images only. False positives from normal images are not included in these two figures.
- **Normal false-positive area fraction:** average `predicted_foreground_pixels / image_pixels` over normal images. This is a pixel-area measure, not the fraction of normal images classified defective.

The threshold objective and test metrics therefore describe candidate masks independently from image-level precision/recall. The low measured pixel recall is a material limitation; section 9 gives the values.

## 7. OpenGL and ICG implementation

### Rendering and coordinate systems

[viewer.cpp](src/viewer.cpp) uses GLFW, GLEW and Dear ImGui. It requests an **OpenGL 3.3 core** context and GLSL `#version 330 core`. The saved run reports `3.3.0 Core Profile Context 24.10.38.09.241016` on an AMD Radeon 660M; the raw JSON renderer string is `AMD RadeonT 660M`. GLEW is the active loader, including for the ImGui backend through the CMake custom-loader definition and forced GLEW header include.

The canvas is rendered directly with OpenGL. `Renderer` converts BGR/gray inputs to contiguous RGB, uploads `GL_RGB8` textures with byte alignment 1 and row length 0, uses linear filtering and clamps texture edges. A VAO/VBO holds six vertices for two textured triangles. Candidate contours, boxes, ground-truth boundaries and the ROI use `GL_LINES`. Texture conversion/upload happens when the source or display mode changes. Dear ImGui draws controls and status panels.

| Coordinate space | Convention and use |
|---|---|
| Image | Native-image coordinates; origin at top left, x right, y down; contours and ROI use this space. |
| Logical window | GLFW/ImGui mouse and canvas coordinates; origin at top left. The fixed control sidebar is 340 units wide. |
| Framebuffer | Physical pixels, which can differ from logical window size under display scaling; used by `glViewport`, `glScissor` and framebuffer capture. |
| Shader clip space | Logical screen position maps to `(2x/windowWidth − 1, 1 − 2y/windowHeight)`. |

The initial window is `1280 × 850` with a `700 × 500` minimum. The drawable image rectangle has 12-unit margins after the sidebar. The model-to-window transform is:

~~~text
M = T(canvasCentre + pan) × S(fit × zoom) × R(angle) × T(−imageWidth/2, −imageHeight/2)

rotatedWidth  = |cos(angle)| × imageWidth  + |sin(angle)| × imageHeight
rotatedHeight = |sin(angle)| × imageWidth  + |cos(angle)| × imageHeight
fit = min(canvasWidth / rotatedWidth, canvasHeight / rotatedHeight)
~~~

CPU `gfx::Mat3` values are row-major and converted to column-major floats for the shader uniform. With image y pointing down, a positive rotation angle appears clockwise. The same transform moves the texture and all overlays. A uniform scale preserves the displayed native-image aspect ratio; the earlier fixed working-grid resize is an analysis operation. Resizing recomputes fit from the logical canvas and rotated bounds. The framebuffer viewport and sidebar scissor use the actual framebuffer/window scale, so display scaling is handled separately.

### Inverse mapping, picking and clipping

The inverse of `M` maps the mouse position back into image coordinates. `pick()` tests actual candidate contours with OpenCV `pointPolygonTest >= 0`, from the last region to the first, including their boundaries. With ROI clipping enabled, a point outside the ROI cannot select a region. The ray-crossing `gfx::pointInPolygon()` helper is exercised by tests but is **not the viewer's picking implementation**.

`gfx::liangBarsky()` ([geometry.cpp](src/geometry.cpp)) clips each candidate contour/box segment against the axis-aligned image-space ROI using its four parametric inequalities. Parallel segments use tolerance `1e-12`; boundary, tangent and degenerate cases have tests. The clipped endpoints are then transformed for display. This differs from the OpenGL scissor, which restricts canvas drawing to the framebuffer area beyond the sidebar.

ROI clipping applies to candidate contours **and boxes**. Ground-truth contours remain visible for comparison and are not clipped by the ROI; all canvas drawing still obeys the sidebar scissor. Neither clipping operation changes the underlying prediction mask.

### Actual controls

These controls are wired in the current source. Saved smoke evidence verifies the rendering/math paths programmatically; physical gestures and native dialogs remain manually unverified.

| Action | Control and actual effect |
|---|---|
| Open image / video | Sidebar buttons use Windows `GetOpenFileNameA`. Image filters include PNG/JPG/JPEG/BMP/TIF; video filters include AVI/MP4/MOV/MKV, with an all-files option. |
| Browse dataset | Dropdown plus Previous/Next over recursively discovered numeric PNG image filenames. |
| Drag and drop | Uses the first dropped file; AVI/MP4/MKV/MOV extensions are treated as video, otherwise as an image. |
| Change display | Original, Processed or Predicted overlay. Processed is the CLAHE grayscale result resized to native dimensions with linear interpolation. |
| Zoom at mouse | Wheel multiplies zoom by `1.15^wheel`, clamped to `0.2–6`, and adjusts pan to keep the cursor's image point fixed. |
| Pan | Right-button or middle-button drag in the canvas. |
| Set transforms | Zoom slider `0.2–6` and rotation slider `−180°–180°`. Rotation also changes the fitted rotated bounds. |
| Reset / fit | Restores zoom 1, angle 0 and zero pan; clears ROI and selection. |
| Pick region | Left click on a candidate contour interior/boundary; shows the selected region's box and pixel area. |
| Define ROI | Enable ROI mode or hold Shift, then left-drag. Inverse-mapped coordinates are clamped to image bounds. |
| Clear / clip ROI | Clear ROI button removes it; “Liang-Barsky clip at ROI” toggles candidate segment clipping and picking restriction. |
| Overlay options | Toggle boxes and explicitly labelled ground truth. Ground truth is disabled without a paired mask and for video frames. |
| Playback | Play/Pause; Step frame advances once and pauses; Rewind seeks to frame zero and pauses. |
| Team details | About panel contains the institute and all three team members. |
| Close | Standard window close control. |

There are no application-defined Space/Escape/arrow/reset shortcuts. Shift is the explicit keyboard modifier used for ROI drawing. The worker can seek to a requested frame index, but the current UI exposes rewind, not an arbitrary seek slider.

## 8. Integration and video

### Requests, worker and synchronization

[main.cpp](src/main.cpp) dispatches CLI modes; [integration.cpp](src/integration.cpp) owns decoding and analysis. OpenCV is set to one analysis thread. The `app::Processor` worker owns its saved `Model` and `VideoCapture`; model load failure is reported instead of triggering training.

A mutex and condition variable protect **one replaceable pending job** and **one completed result**, alongside any in-flight job. Each request receives an increasing token. A new request replaces the pending job and clears an older completed result. Already running work is allowed to finish, but its result is discarded if its token is obsolete. `take()` accepts only a result matching the latest request. This bounds the waiting queue and prevents stale completion from replacing a newer request.

A packet carries the decoded image, analysis result, source identity, frame index, dimensions, source FPS and decode timing. The viewer commits a matching packet together and invalidates its texture. It can retain the previous complete packet while loading or after failure; the “Displayed” source and pending/error status identify what is actually visible. It does not display a new frame with an old frame's contours.

Images use `imread(IMREAD_COLOR)`. Video uses OpenCV `CAP_FFMPEG`; the worker records the frame index before reading and supports open, next and seek jobs. Primary decode/analysis runs off the render thread. Texture conversion/upload and an occasional ground-truth-mask load/contour extraction still occur on the UI thread. Ground-truth contours are cached for the displayed image when requested and never passed to the prediction worker.

### Playback and demo scope

Playback submits the next frame only after the preceding request has completed and a source-FPS interval has elapsed. It does not grow an unbounded frame queue. Because the interval follows completion, playback can be slower than real time when analysis is costly. It pauses at the last advertised frame; arbitrary codecs, files with unknown frame counts and live streams were not verified. There is no audio playback, temporal classifier or tracking with persistent defect IDs.

`makeDemo()` sorts canonical test paths (the same as numeric ID order for this dataset) and writes the first **24 still images** as MJPEG AVI at **8 FPS**, resizing to the first image's dimensions (`230 × 636`). [demo_frames.csv](outputs/demo_frames.csv) identifies frames `20000–20023`; [demo_sequence.avi](outputs/demo_sequence.avi) is approximately three seconds long.

This is a decoded-video integration demonstration made from dataset stills. It is **not independent natural-video evaluation** and provides no video accuracy result. The runtime check compares six decoded frames with a separate decoder and saved-model analysis; maximum score difference is zero. A separate seek to frame 2 matches the reference decoded pixels. The graphics smoke run displayed four successive video frames. These establish decoding/synchronization consistency, not temporal detection quality.

### Error handling

Missing/undecodable inputs, failed model/configuration loads and unavailable video opens produce explicit errors. Worker exceptions carry the request token; the viewer retains the last displayed packet, shows the error and stops playback. A missing/mismatched paired mask disables or reports the ground-truth display. Shader compilation/linking, OpenGL context/rendering failures and singular inverse transforms are also checked. Top-level CLI exceptions return a nonzero exit status. Runtime tests cover missing input, invalid decoding, stale requests, image-only inference and video consistency; they do not cover every codec or GUI-dialog failure.


## 9. Saved experimental results

All values in this section come from the existing saved run, not new training/evaluation during this audit. Fractions are shown to six decimals where useful; linked JSON/CSV files retain their recorded precision. **Normal is class 0; defective is class 1.**

`metrics()` in [evaluation.cpp](src/evaluation.cpp) computes accuracy `(TP+TN)/N`, defect precision `TP/(TP+FP)`, recall `TP/(TP+FN)` and F1 `2PR/(P+R)`, with zero-denominator guards. It sorts scores descending and groups ties. ROC-AUC uses trapezoidal integration; average precision sums each recall increment times precision after its score group. These score-ranking measures do not depend on the chosen classification threshold.

### Validation comparison

[model_comparison.csv](outputs/model_comparison.csv) evaluates all candidates on **466 validation images: 417 normal and 49 defective**. “Ablated” means `without_residual_texture`: columns 51–109 are zero. Thresholds are validation-selected. Accuracies below are derived from each saved FP/FN count as `(466 − FP − FN)/466`; the other columns are saved metrics.

| Features / model | Parameter | Threshold | Accuracy % | Defect precision | Recall | F1 | F2 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority normal | Constant 0 | 1 | 89.4850 | 0 | 0 | 0 | 0 | 0.500000 | 0.105150 |
| Full RF | Depth 10 | 0.235 | 94.8498 | 0.765957 | 0.734694 | 0.750000 | 0.740741 | 0.916703 | 0.815625 |
| Full RF | Depth 16 | 0.205 | 94.2060 | 0.711538 | 0.755102 | 0.732673 | 0.745968 | 0.915529 | 0.820248 |
| **Full SVM — selected** | **C=1** | **−0.0818077** | **93.9914** | **0.684211** | **0.795918** | **0.735849** | **0.770751** | **0.911614** | **0.749882** |
| Full SVM | C=10 | −0.432047 | 90.9871 | 0.546667 | 0.836735 | 0.661290 | 0.756458 | 0.892625 | 0.737628 |
| Ablated RF | Depth 10 | 0.310 | 94.2060 | 0.789474 | 0.612245 | 0.689655 | 0.641026 | 0.863823 | 0.709755 |
| Ablated RF | Depth 16 | 0.125 | 84.9785 | 0.387097 | 0.734694 | 0.507042 | 0.622837 | 0.852861 | 0.699876 |
| Ablated SVM | C=1 | −0.265051 | 90.3433 | 0.531250 | 0.693878 | 0.601770 | 0.653846 | 0.890912 | 0.710079 |
| Ablated SVM | C=10 | −0.288731 | 88.1974 | 0.462500 | 0.755102 | 0.573643 | 0.670290 | 0.879655 | 0.690735 |

The selected SVM has validation confusion counts **TN=399, FP=18, FN=10, TP=39**. Its higher F2 drives selection even though RF depth 10 has higher accuracy and F1. This comparison does not establish that SVM is best under every objective.

### Held-out test classification

[final_test.json](outputs/final_test.json), [test_predictions.csv](outputs/test_predictions.csv) and [confusion_matrix.csv](outputs/confusion_matrix.csv) record **1,004 test images: 894 normal and 110 defective**, using the saved full SVM C=1 and threshold **−0.081807687878608704**.

| Metric | Selected SVM | Always-normal baseline |
|---|---:|---:|
| Accuracy | 0.930279 (93.0279%) | 0.890438 (89.0438%) |
| Defect precision | 0.647059 | 0 |
| Defect recall | 0.800000 | 0 |
| Defect F1 | 0.715447 | 0 |
| Defect F2 | 0.763889 | 0 |
| ROC-AUC | 0.938316 | 0.500000 |
| Average precision | 0.806855 | 0.109562 |

Confusion matrix: **rows = actual class; columns = predicted class**, ordered normal then defective.

| Actual \ Predicted | Normal (0) | Defective (1) |
|---|---:|---:|
| Normal (0) | **846 TN** | **48 FP** |
| Defective (1) | **22 FN** | **88 TP** |

The image-level false-positive rate on normals is `48/894 = 5.3691%`. The classifier misses `22/110 = 20%` of defective test images. The majority baseline has TN=894, FP=0, FN=110, TP=0; its high accuracy reflects imbalance while finding no defects. No final-test RF comparison was saved.

### Candidate-mask results

Native-size test localization results are stored in [test_localization.csv](outputs/test_localization.csv) and [final_test.json](outputs/final_test.json), using threshold 30 and minimum area 24.

| Metric | Saved test result |
|---|---:|
| Defective images / normal images | 110 / 894 |
| Mean defective IoU | 0.204163 |
| Mean defective Dice | 0.295894 |
| Aggregated defective-image pixel precision | 0.924821 |
| Aggregated defective-image pixel recall | 0.153852 |
| Mean normal false-positive area fraction | 0.000279331637 (**0.0279332%**) |

The selected validation setting had mean defective Dice **0.401768** and normal foreground fraction **0.000298453**. High defective-image pixel precision together with low pixel recall means the method highlights a small subset of annotated defects. Normal-image pixel errors are reported separately, as explained in section 6; the pixel precision above does not include them.

### Training and analysis timings

The following timings have different scopes and must not be added or compared as if all measured a complete application frame.

| Measurement | Saved value | Scope |
|---|---:|---|
| Development feature extraction | 43.991022 ms/image | Feature construction after image decode; [training_summary.json](outputs/training_summary.json). |
| Development total | 281.221722 s | Dataset audit/manifest, features, eight model fits, validation/localization search, save/reload and reconstruction checks; sampled before final test. Not just SVM fit time. |
| Test feature extraction | 40.635093 ms/image | Features only; `imread` is outside this timer. |
| Test batch prediction | 0.039190 ms/image | Batch normalization and scoring over 1,004 rows; not complete per-image application inference. |
| Test feature + prediction throughput | 24.585559 images/s | Reciprocal of those two stages; excludes decoding, localization and rendering. |
| Test localization evaluation | 14.163188 ms/image | Includes image **and mask decode**, GT thresholding, localization, metric computation and per-image CSV work inside the evaluation loop. Not isolated localization inference. |

The model-fit durations from the validation CSV are below. Their timer encloses `Model::train()`, including scaler fitting/normalization and SVM margin orientation, and excludes validation scoring/threshold search:

| Candidate | Full features, ms | Ablated features, ms |
|---|---:|---:|
| RF depth 10 | 3986.85 | 3420.97 |
| RF depth 16 | 4231.63 | 3557.12 |
| SVM C=1 | 170.003 | 185.533 |
| SVM C=10 | 127.055 | 146.070 |

The majority baseline is not trained and records 0 ms. Ablation does not demonstrate a faster feature extractor: response maps are still computed before their feature columns are zeroed. In the viewer, “Localization + processed view” times `Model::analyze()` localization/region construction plus another preprocessing/display preparation; it has a different scope from the evaluation-loop timing above.

### Rendering versus analysis

The workspace [graphics_smoke.json](outputs/graphics_smoke.json) records:

| Measurement | Saved value |
|---|---:|
| Rendered loop frames | 582 |
| Render work mean / p95 | 0.549484 ms / 0.485500 ms |
| `1000 / mean render work` | 1819.888568 frames/s |
| Mean analysis time | 53.451060 ms |
| Analyses in this smoke sample | 5: one image and four video frames |
| Video frames displayed | 4 |
| Genuine framebuffer screenshots | 6 |
| Framebuffer/source pixel samples | 9 |
| Maximum observed sampled channel error | 1 |

The render timer covers canvas texture work when needed, overlay construction/drawing, ImGui draw submission and **`glFinish()` in smoke mode**. It starts after UI/input handling and optional cached GT loading, and excludes buffer swapping/vsync, event polling, worker decode and analysis. Normal mode enables vsync and does not force `glFinish()` each frame.

The reported 1819.89 value is render-work throughput, not visible display FPS, analyzed-video FPS or sustained application throughput. Many rendered loops reuse the same analyzed image. The small analysis sample implies approximately **18.71 analyses/s** by reciprocal, excluding decode; it is not a sustained end-to-end video benchmark. No full end-to-end video FPS, isolated localizer inference benchmark or peak application RAM measurement is available.

[portable_package_verification.json](outputs/portable_package_verification.json) contains a **separate extracted-package run** with different timings (448 rendered frames). Those figures should not be mixed with this workspace run.

### Existing success/failure examples and captures

The offline example panels are original image, ground truth and candidate overlay. TP/TN/FP/FN filenames refer to **image classification**, not pixel segmentation success. The first two test IDs for each outcome were saved; these are illustrative cases, not an extra evaluation set.

| Classification outcome | Existing panels |
|---|---|
| True positive | [TP 20042](outputs/examples/TP_20042.png), [TP 20056](outputs/examples/TP_20056.png) |
| True negative | [TN 20000](outputs/examples/TN_20000.png), [TN 20001](outputs/examples/TN_20001.png) |
| False positive | [FP 20008](outputs/examples/FP_20008.png), [FP 20027](outputs/examples/FP_20027.png) |
| False negative | [FN 20068](outputs/examples/FN_20068.png), [FN 20095](outputs/examples/FN_20095.png) |

The six saved OpenGL framebuffer captures are [original](outputs/screenshots/01_original.png), [processed](outputs/screenshots/02_processed.png), [rotation/ROI/overlay](outputs/screenshots/03_rotation_roi_overlay.png), [resized overlay](outputs/screenshots/04_resized_overlay.png), [video frame](outputs/screenshots/05_video_frame.png) and [About/team](outputs/screenshots/06_about_team.png). They were generated by automated smoke mode, not physical mouse gestures.

## 10. Build and run

### Tested environment and dependencies

[environment.json](outputs/environment.json) records Windows 11 Home Single Language `10.0.26200`, AMD Ryzen 5 6600H, approximately 16 GB installed RAM, AMD Radeon 660M and:

| Dependency | Tested version / use |
|---|---|
| MSYS2 MinGW-w64 GCC | 16.2.0, C++17 Release |
| CMake / Ninja | CMake project minimum 3.20; Ninja generator |
| OpenCV | 5.0.0; `core, imgproc, imgcodecs, ml, videoio, geometry` components |
| GLFW | 3.5.1; window, input and OpenGL context |
| GLEW | 2.3.1; OpenGL function loading |
| Dear ImGui | 1.91.9b; GLFW/OpenGL3 backends |
| OpenGL driver | 3.3 core context verified on the saved AMD system |
| Python | Setup scripts use the existing MSYS2 Python; dependency extraction uses `compression.zstd`, requiring Python 3.14+ |
| Other setup tools | Existing MinGW `g++`/`objdump`, curl and the DLL dependencies resolved by the setup script |

Analysis and classical ML run on the CPU; OpenGL accelerates display. There is no CUDA/deep-learning inference implementation. The saved memory figures describe system memory at setup, not peak application memory.

The source and helper scripts were tested with **one Windows 11 x64 development machine**. Windows 10/11 x64 is the portable trial's intended target, but another machine/Windows 10 was not verified. Linux, macOS and MSVC builds were not tested; the native file-picker implementation is Windows-specific. The `MSVC` CMake branch alone is not proof of MSVC compatibility.

The GitHub repository includes source, setup/build scripts, the saved model, dataset provenance, evaluation reports, examples, screenshots and the small demonstration video. Raw dataset images/masks, the dataset ZIP, downloaded dependencies, build executables, caches, logs and the portable trial ZIP are excluded by [.gitignore](.gitignore). A fresh clone therefore requires dependency setup and a build before launching; dataset commands also require the separate dataset preparation step.

Clone the repository in PowerShell, then run the commands below **from its project root**:

~~~powershell
git clone https://github.com/gyanchandra2910/AI-Assisted-Surface-Defect-Inspection.git
Set-Location -LiteralPath .\AI-Assisted-Surface-Defect-Inspection
~~~

For an existing checkout, open PowerShell in that directory. Commands are checked against the scripts/CLI; only those explicitly listed in section 12 were rerun for the 18 October documentation audit.

### Use the existing saved model

With the current executable and dependencies present:

~~~powershell
.\tools\run.ps1 help
.\tools\run.ps1 view
.\tools\run.ps1 predict --image data\KolektorSDD2\test\20042.png
~~~

For a separate image:

~~~powershell
.\tools\run.ps1 view --image 'C:\path\to\surface.png' --model outputs\model
~~~

Replace the example image path with a real image. `run.ps1` sets the dependency DLL path and runs `build/bin/inspect.exe` from the project root. **Launching `view` or `predict` never trains.** A valid `classifier.xml` and `pipeline.yml` in the model directory are required.

CLI defaults are `--data data/KolektorSDD2`, `--output outputs` and `--model outputs/model`. The mode must come first, followed by options. `--output` does **not** change `--model` automatically. In particular, to use a model trained into another output directory, pass its model path explicitly.

### Reproduce dependency setup, data preparation and build

These commands download/setup or rebuild project inputs. They were **not rerun** for this documentation task. They assume an existing compatible MSYS2 MinGW-w64 compiler, `objdump` and Python at `C:\msys64\mingw64\bin`; the dependency helper does not install the compiler itself.

~~~powershell
& 'C:\msys64\mingw64\bin\python.exe' tools\setup_dependencies.py
& 'C:\msys64\mingw64\bin\python.exe' tools\prepare_dataset.py
.\tools\build.ps1
~~~

[setup_dependencies.py](tools/setup_dependencies.py) reads/caches the MSYS2 package database, verifies package SHA-256 values, extracts dependencies under `.deps/mingw64`, resolves DLL imports and obtains ImGui 1.91.9b. It can reuse installed MinGW runtime DLLs. Package names are recorded in `.deps/versions.json`; this is not a fully version-pinned dependency lock.

[prepare_dataset.py](tools/prepare_dataset.py) downloads or reuses the ZIP, checks its CRC, rejects unsafe archive paths and extracts under `data/KolektorSDD2`. Its default download uses four range connections. `--connections 1` uses the single-connection/resume path. Existing extracted files of the expected size can be skipped; this is not a new content-hash verification of each existing extracted file. `--inspect-only` still checks the archive and writes metadata, so it is not a read-only inspection command.

[build.ps1](tools/build.ps1) configures Release/Ninja, builds with two jobs and runs CTest. Its actual CMake equivalents are:

~~~powershell
$env:PATH = "$PWD\.deps\mingw64\bin;C:\msys64\mingw64\bin;$env:PATH"
& .\.deps\mingw64\bin\cmake.exe -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$PWD/.deps/mingw64" -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe
& .\.deps\mingw64\bin\cmake.exe --build build --parallel 2
& .\.deps\mingw64\bin\ctest.exe --test-dir build --output-on-failure
~~~

Targets are `analysis` and `graphics` libraries, the `inspect` application and `core_tests`. CTest registers `core_tests`. The checked build uses the MinGW ABI consistently; replacing libraries with an MSVC OpenCV build is not the tested setup.

### Dataset audit, training and evaluation

The Python download/extraction step and the C++ `prepare` command have different jobs. The C++ command decodes/validates the extracted dataset and writes a manifest/audit. The following commands use separate output directories to preserve existing results:

~~~powershell
.\tools\run.ps1 prepare --output runs\dataset_audit

# Optional, expensive: fits and compares models, selects on validation,
# saves runs\reproduce\model, then evaluates the selected model on test.
.\tools\run.ps1 train --output runs\reproduce

# Optional full test evaluation of the already saved model.
.\tools\run.ps1 evaluate --model outputs\model --output runs\saved_model_eval

# View a separately reproduced model explicitly.
.\tools\run.ps1 view --model runs\reproduce\model
~~~

Do not run training/evaluation just to open the viewer. The default output directory is the existing `outputs` directory; rerunning commands without another `--output` can overwrite its experiment files.

### Existing checks and video smoke mode

For the lightweight core executable, set the DLL path as above, then run:

~~~powershell
& .\build\bin\core_tests.exe
~~~

For new runtime/graphics evidence in a separate directory:

~~~powershell
.\tools\run.ps1 demo-video --output runs\smoke
.\tools\run.ps1 verify --output runs\smoke --model outputs\model
.\tools\run.ps1 view --smoke --output runs\smoke --model outputs\model --image data\KolektorSDD2\test\20042.png
~~~

`verify` and graphics smoke mode need `runs/smoke/demo_sequence.avi`, which the first command produces. The image 20042 has candidate regions used by the picking check. Smoke mode sets transformations/ROI, resizes the window, renders/captures six stages, exercises video and exits automatically, with a 90-second timeout. A working desktop/OpenGL context is required.

| Mode | Main input | Files written / behaviour |
|---|---|---|
| `help` / `--help` | None | Prints usage. |
| `prepare` | Extracted `--data` | `manifest.csv` and `dataset_audit.txt` under `--output`; no download. |
| `train` | Extracted canonical train/test | Development split/comparison, selected `model/`, validation/localization records and final-test outputs under `--output`. |
| `evaluate` | Dataset + `--model` | Recreates final-test evaluation outputs/examples under `--output`; no training. |
| `predict` | `--image` + `--model` | Prints classification, score, threshold and candidate count; no result files. |
| `view` | Saved model; optional `--image` | Interactive viewer; creates the screenshots directory. `--smoke` writes six screenshots and `graphics_smoke.json`. |
| `demo-video` | Numeric test images | `demo_sequence.avi` and `demo_frames.csv` under `--output`. |
| `verify` | Model, dataset and `--output/demo_sequence.avi` | `runtime_checks.json`; temporary invalid/copy image fixtures are removed. |

Non-view modes do not initialize GLFW, although the shared `inspect.exe` still links graphics DLLs.

### Existing trial ZIP

The previously created local archive `share/AI_Surface_Defect_Inspection_Trial.zip` is **101,638,109 bytes** (about 96.9 MiB). It is excluded from Git and is not included in a clone. Its saved SHA-256 is:

~~~text
d32403a3904d479cbf37c64fa61323f3b124f1dd830d3eb827a9af53fa126847
~~~

If you separately receive the trial archive, extract all files, then run `AI_Surface_Defect_Inspection_Trial/START_APP.cmd`; `RUN_CHECKS.cmd` runs its included checks. It contains binaries, DLLs, source, the saved model/results and 29 paired samples (25 normal, four defective), rather than the full dataset. Its isolated-DLL-path verification is recorded in the local `outputs/portable_package_check.log` (excluded from Git) and [portable_package_verification.json](outputs/portable_package_verification.json). This README audit does **not** rebuild or update the existing ZIP; its bundled documentation is the earlier snapshot.


## 11. Project structure

| Path | Purpose |
|---|---|
| [CMakeLists.txt](CMakeLists.txt) | C++17 targets, OpenCV/OpenGL dependencies and ImGui backend configuration. |
| [src/pipeline.hpp](src/pipeline.hpp), [src/pipeline.cpp](src/pipeline.cpp) | Configuration, preprocessing, descriptors, classification, candidate masks, model persistence and dataset validation. |
| [src/evaluation.cpp](src/evaluation.cpp) | Split construction, training/selection, classification/localization metrics and exported examples. |
| [src/geometry.hpp](src/geometry.hpp), [src/geometry.cpp](src/geometry.cpp) | 3×3 transforms/inverse, Liang–Barsky clipping and geometry helpers. |
| [src/viewer.cpp](src/viewer.cpp) | OpenGL renderer, ImGui controls, picking, ROI and graphics smoke mode. |
| [src/integration.hpp](src/integration.hpp), [src/integration.cpp](src/integration.cpp) | Worker packets/jobs, image/video decoding, demo creation and runtime checks. |
| [src/main.cpp](src/main.cpp) | CLI argument parsing and mode dispatch. |
| [tests/core_tests.cpp](tests/core_tests.cpp) | Existing geometry, consistency, decoding and descriptor tests. |
| [tools/build.ps1](tools/build.ps1), [tools/run.ps1](tools/run.ps1) | Current Windows build/test and launch wrappers. |
| [tools/setup_dependencies.py](tools/setup_dependencies.py), [tools/prepare_dataset.py](tools/prepare_dataset.py) | Dependency setup and dataset download/extraction. |
| [tools/package_project.py](tools/package_project.py) | Existing portable-trial packager. |
| [data/source.json](data/source.json), [outputs/dataset_source.json](outputs/dataset_source.json) | Dataset source/archive provenance. Canonical images/masks are under `data/KolektorSDD2/train` and `test`. |
| [outputs/manifest.csv](outputs/manifest.csv), [outputs/development_split.csv](outputs/development_split.csv), [outputs/dataset_audit.txt](outputs/dataset_audit.txt) | Dataset identities, class labels, fit/validation assignment and audit. |
| [outputs/model/pipeline.yml](outputs/model/pipeline.yml), [outputs/model/classifier.xml](outputs/model/classifier.xml) | Active preprocessing/scaler/decision configuration and selected SVM. |
| [outputs/training_summary.json](outputs/training_summary.json), [outputs/model_comparison.csv](outputs/model_comparison.csv), [outputs/localization_validation.csv](outputs/localization_validation.csv) | Selection, reload checks, model comparison and localization search. |
| [outputs/final_test.json](outputs/final_test.json), [outputs/test_predictions.csv](outputs/test_predictions.csv), [outputs/test_localization.csv](outputs/test_localization.csv), [outputs/confusion_matrix.csv](outputs/confusion_matrix.csv) | Test summary and per-image classification/localization evidence. |
| [outputs/environment.json](outputs/environment.json), [outputs/runtime_checks.json](outputs/runtime_checks.json), [outputs/graphics_smoke.json](outputs/graphics_smoke.json) | Saved environment, integration and rendering evidence. |
| [outputs/examples](outputs/examples), [outputs/screenshots](outputs/screenshots) | Saved classification outcome panels and six OpenGL captures. |
| [outputs/demo_sequence.avi](outputs/demo_sequence.avi), [outputs/demo_frames.csv](outputs/demo_frames.csv) | Still-image video demo and its frame provenance. |
| Local `share/AI_Surface_Defect_Inspection_Trial.zip` (excluded from Git), [outputs/portable_package_verification.json](outputs/portable_package_verification.json) | Existing portable trial and separate verification record. |

The downloaded archives, dependency tree and generated build tree are omitted from this compact listing. [.gitignore](.gitignore) excludes large/raw/generated content as configured; their presence in this working directory is not evidence that every file is tracked in version control.

## 12. Verification, limitations and attribution

### What was actually verified

| Evidence | What it supports |
|---|---|
| Existing local CTest log, `build/Testing/Temporary/LastTest.log` (excluded from Git) | `core_tests` passed on 13 October 2026. |
| Core test source and fresh direct executable run on 18 October 2026 | 500 inverse/picking cases; clipping parallel/boundary/tangent/degenerate cases plus 500 reversal cases; stale tags; decoding; feature schema/ablation/scaling; localization dimensions. |
| [training_summary.json](outputs/training_summary.json) | Eight-candidate development experiment completed; selected model saved/reloaded with maximum validation score difference zero; no validation refit. |
| [runtime_checks.json](outputs/runtime_checks.json) | Missing/invalid input rejection, latest request wins, image-only saved-model inference, six decoded-video frames and seek agreement, queue capacity one. |
| [graphics_smoke.json](outputs/graphics_smoke.json) and saved captures | Real OpenGL framebuffer rendering, programmatically set transforms/ROI/picking, resize and video stages; nine texture alignment samples with maximum channel error one. |
| Existing portable verification/log | 399 file hashes verified, 144 bundled DLLs, 29 paired samples, successful core/runtime/graphics checks with installed MinGW excluded from the test PATH. This was a second run on the same development machine. |
| Current documentation audit | Read source/configuration/models/scripts/results; cross-checked manifest/split counts, labels, file presence, recorded hashes and prediction outcomes; compared both extra raw copies byte-for-byte with their canonical counterparts. No full dataset decode/hash pass was repeated. |
| Current source/build/model comparison | CMake, source, test files, both build executables and saved model/configuration match the corresponding contents of the previously tested trial ZIP by SHA-256. |

The lightweight commands actually rerun during this documentation audit were `core_tests.exe`, CLI `help` and saved-model `predict` for `test/20042.png`, with the dependency DLL path set. They passed. The prediction printed:

~~~text
defective, score=0.391052, threshold=-0.0818077, candidate regions=10
~~~

These checks did not retrain, evaluate the whole dataset or overwrite saved results. CTest was not rerun, preserving its existing log; the core executable was invoked directly. Existing training, runtime, screenshots and graphics measurements remain saved evidence from the earlier runs.

### Material limitations and verification gaps

- **Classification errors remain:** test precision is about 64.71%, recall 80%, with 48 false positives and 22 missed defective images. The score is uncalibrated; there is no confidence-probability calibration or independent industrial deployment study.
- **Localization is incomplete:** mean defective Dice is 0.295894 and defective-image pixel recall is 0.153852. Candidate contours are suspected regions from a morphological method, not confirmed defect boundaries or physical measurements.
- **ROI scope is visual:** it clips candidate segments and restricts picking; it does not change classifier features, candidate-mask computation or evaluation.
- **Feature geometry is fixed:** analysis resizes to 224×608, independently of native aspect ratio. Integer spatial cells omit two working-grid columns and ranked patches leave edge strips. External contours do not preserve holes as separate region boundaries.
- **Video accuracy is unavailable:** the tested AVI is made from test stills. Decoding/score consistency and a few displayed frames were verified; independent natural-video detection, live streams, arbitrary codecs and sustained end-to-end video FPS were not.
- **Manual GUI verification is incomplete:** the smoke run sets states programmatically. Native file dialogs, actual mouse/Shift gestures and manual widget operation have not been verified. The ANSI Windows picker also has no recorded Unicode-path test.
- **Measurement coverage is limited:** saved RF results are validation-only; the selected SVM is the sole persisted candidate. There is no isolated localizer inference time, peak process RAM record, independent machine test or complete application FPS measurement.
- **Compatibility is demonstrated on one machine:** other Windows versions, MSVC, Linux/macOS and other graphics drivers are unverified. Existing portable checks isolate DLL discovery but do not constitute a second-machine test.
- **Audit wording needs context:** the historical extra-GT-copy message compares the mask copy against an image-only hash map, not its canonical mask. Current direct byte comparison resolves both extra copies as identical to their canonical files (section 4); no underlying image/mask files or old audit output were modified.

No exact content-hash overlap is recorded between fit/validation/test. Related production samples can still share acquisition conditions; the available records do not establish independence by production batch or acquisition time.

### Attribution and available licences

Dataset provenance is recorded in [data/source.json](data/source.json) and [outputs/dataset_source.json](outputs/dataset_source.json). The project uses **KolektorSDD2**, industrial images associated with Kolektor Group and distributed through ViCoS. The recorded dataset licence is **CC BY-NC-SA 4.0**, including attribution, noncommercial and share-alike terms. Consult the [KolektorSDD2 resource page](https://www.vicos.si/resources/kolektorsdd2/) and the dataset notice retained in the existing trial package before further redistribution.

Dataset publication attribution: **Jakob Božič, Domen Tabernik and Danijel Skočaj (2021), “Mixed supervision for surface-defect detection: from weakly to fully supervised learning,” Computers in Industry** ([publication page](https://prints.vicos.si/publications/385)). The title/authors/source are provided as attribution; this project does not claim to reproduce that paper's neural method.

The following facts come from locally available dependency metadata/licence files; this documentation task did not perform a new online licence review.

| Library / tooling | Local licence information / attribution |
|---|---|
| [OpenCV](https://opencv.org/) 5.0.0 | MSYS2 package metadata records `Apache-2.0`. |
| [GLFW](https://www.glfw.org/) 3.5.1 | `Zlib`; licence retained under `.deps/mingw64/share/licenses/glfw/LICENSE`. |
| [GLEW](https://glew.sourceforge.net/) 2.3.1 | Package metadata lists `Modified BSD, MIT, GPL` for included components; see `.deps/mingw64/share/licenses/glew/LICENSE` rather than assigning one licence to the whole distribution. |
| [Dear ImGui](https://github.com/ocornut/imgui/tree/v1.91.9b) 1.91.9b | MIT; Omar Cornut, 2014–2025, in `.deps/imgui-1.91.9b/LICENSE.txt`. |
| [FFmpeg](https://ffmpeg.org/) via OpenCV video I/O | Cached MSYS2 package metadata records `GPL-3.0-or-later` for the included package. |
| [MSYS2](https://www.msys2.org/) and transitive runtime libraries | Dependency provenance is available in cached package metadata; licences vary by package. |

Other notices are present under `.deps/mingw64/share/licenses` and the installed MinGW runtime's licence directory. The existing trial includes `third_party/licenses` and `DEPENDENCY_SOURCES.json`. These locations and the table are an inventory of available information, not a complete verification of every bundled DLL's redistribution obligations; that full transitive dependency/licence check remains unresolved.

There is currently **no top-level project LICENSE file** assigning a licence to the team's own source. Dataset and third-party licences do not assign one to that source. No project licence or individual ownership/contribution claims are invented here.
