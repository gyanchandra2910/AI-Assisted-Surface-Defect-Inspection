# AI-Assisted Surface Defect Detection and Interactive Visualization

**IIITDM Kancheepuram · Interactive Computer Graphics · Theme 6**

A C++ application for industrial surface inspection using classical computer vision, machine learning and an interactive OpenGL viewer.

**[Read the detailed documentation →](README_DETAILED.md)** — algorithms, feature schema, dataset splits, controls, experiments, verification and licences.

## Features

- Image-level normal/defective classification with a saved RBF SVM.
- Separate candidate-region masks, contours, bounding boxes and pixel areas.
- Original, CLAHE-processed and overlay views with zoom, pan, rotation, ROI clipping and region picking.
- Image browsing and video loading, playback, frame stepping and rewind.
- Saved evaluations, success/failure examples and OpenGL screenshots.

**Stack:** C++17, OpenCV 5, OpenGL 3.3, GLFW, GLEW and Dear ImGui.

## Quick start

Tested on Windows 11 x64 with MSYS2 MinGW64 GCC. Setup requires an existing compiler, `objdump` and Python 3.14+ under `C:\msys64\mingw64\bin`, plus an OpenGL 3.3-capable driver.

Run in PowerShell:

~~~powershell
git clone https://github.com/gyanchandra2910/AI-Assisted-Surface-Defect-Inspection.git
Set-Location -LiteralPath .\AI-Assisted-Surface-Defect-Inspection

& 'C:\msys64\mingw64\bin\python.exe' tools\setup_dependencies.py
& 'C:\msys64\mingw64\bin\python.exe' tools\prepare_dataset.py
.\tools\build.ps1
.\tools\run.ps1 view
~~~

The saved model is included; opening the viewer does **not** train it. Raw data, downloaded libraries, build binaries and trial ZIPs are excluded from Git. See the [full setup and CLI guide](README_DETAILED.md#10-build-and-run) for other commands.

## Saved results

KolektorSDD2 held-out test set: **1,004 images**, selected full-feature SVM.

| Accuracy | Defect precision | Defect recall | Defect F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 93.03% | 64.71% | 80.00% | 0.7154 | 0.9383 |

See [test results](outputs/final_test.json), [example panels](outputs/examples), [viewer screenshots](outputs/screenshots) and the [full comparison](README_DETAILED.md#9-saved-experimental-results).

Candidate localization is independent of classification and has mean defective-image Dice **0.2959**. ROI affects visualization/picking only. The video demo is made from dataset stills; independent video accuracy was not evaluated.

## Team

| Member | Roll number |
|---|---|
| Gyan Chandra | CS23I1053 |
| Vaibhav | CS23I1040 |
| Vishal Singh | CS23I1039 |

## Dataset and attribution

[KolektorSDD2 — ViCoS](https://www.vicos.si/resources/kolektorsdd2/) is recorded as **CC BY-NC-SA 4.0**. Dataset/publication credits, dependency licences and verification gaps are in the [detailed documentation](README_DETAILED.md#12-verification-limitations-and-attribution). No licence has yet been assigned to the team's own source.
