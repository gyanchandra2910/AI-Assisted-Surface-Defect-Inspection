"""Create a portable Windows x64 trial, including the complete runtime import closure.

The official dataset subset is for demonstration, never a replacement evaluation split.
"""
import collections
import csv
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import tarfile
import uuid
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFIX = ROOT / '.deps' / 'mingw64'
ARCHIVE_ROOT = 'AI_Surface_Defect_Inspection_Trial'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def runtime_closure():
    libraries = {p.name.lower(): p for p in (PREFIX / 'bin').glob('*.dll')}
    system = pathlib.Path('C:/Windows/System32')
    queue = collections.deque((ROOT / 'build' / 'bin').glob('*.exe'))
    seen, selected, windows = set(), {}, set()
    while queue:
        path = queue.popleft()
        key = path.name.lower()
        if key in seen:
            continue
        seen.add(key)
        headers = subprocess.check_output(
            ['C:/msys64/mingw64/bin/objdump.exe', '-p', str(path)], text=True)
        for name in re.findall(r'DLL Name: (.+)', headers):
            name = name.strip()
            key = name.lower()
            # Use the machine's graphics driver, never package Mesa's opengl32.dll.
            # Vulkan is not a built-in Windows DLL, so include its local loader even
            # when this development machine has a driver-installed copy in System32.
            if key == 'opengl32.dll' or key.startswith(('api-ms-', 'ext-ms-')):
                windows.add(name)
            elif key in libraries:
                selected[key] = libraries[key]
                queue.append(libraries[key])
            elif (system / name).exists():
                windows.add(name)
            else:
                raise RuntimeError('Unresolved portable runtime import: ' + name)
    return selected, sorted(windows)


def package_metadata():
    packages, providers = {}, {}
    db = ROOT / '.deps' / 'cache' / 'mingw64.files'
    with tarfile.open(db, 'r:*') as archive:
        for entry in archive:
            if entry.name.endswith('/desc'):
                raw = archive.extractfile(entry).read().decode()
                fields = dict(re.findall(r'%([^%]+)%\n(.*?)(?=\n%|\Z)', raw, re.S))
                name = fields['NAME'].strip()
                packages[name] = {k: v.strip().splitlines() for k, v in fields.items()}
            elif entry.name.endswith('/files'):
                package = re.sub(r'-[^-]+-[^-]+$', '', entry.name.split('/')[0])
                for filename in archive.extractfile(entry).read().decode().splitlines():
                    if filename.startswith('mingw64/bin/') and filename.lower().endswith('.dll'):
                        providers[filename.split('/')[-1].lower()] = package
    return packages, providers


def remove_own_staging(path):
    target = path.resolve()
    share = (ROOT / 'share').resolve()
    if target.parent != share or not target.name.startswith('.package_stage_'):
        raise RuntimeError('Refusing to remove unexpected staging path: ' + str(target))
    shutil.rmtree(target)


def main():
    share = ROOT / 'share'
    share.mkdir(exist_ok=True)
    staging = share / ('.package_stage_' + uuid.uuid4().hex)
    staging.mkdir()
    destination = staging / ARCHIVE_ROOT
    destination.mkdir()
    dlls, windows = runtime_closure()
    for exe in (ROOT / 'build' / 'bin').glob('*.exe'):
        copy(exe, destination / 'bin' / exe.name)
    for path in dlls.values():
        copy(path, destination / 'bin' / path.name)
    for name in ['CMakeLists.txt', '.clang-format', '.gitignore']:
        copy(ROOT / name, destination / name)
    for directory in ['src', 'tests', 'tools', 'outputs']:
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.tmp', '*.log'))

    predictions = {row['id']: row for row in csv.DictReader(
        (ROOT / 'outputs' / 'test_predictions.csv').open())}
    samples = {row['source_image'][:-4] for row in csv.DictReader(
        (ROOT / 'outputs' / 'demo_frames.csv').open())}
    for path in (ROOT / 'outputs' / 'examples').glob('*.png'):
        samples.add(path.stem.split('_', 1)[1])
    manifest = []
    for image_id in sorted(samples):
        for suffix in ['.png', '_GT.png']:
            name = image_id + suffix
            copy(ROOT / 'data' / 'KolektorSDD2' / 'test' / name,
                 destination / 'data' / 'KolektorSDD2' / 'test' / name)
        manifest.append(predictions[image_id])
    with (destination / 'DEMO_SAMPLES.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'label', 'score', 'prediction', 'outcome'])
        writer.writeheader()
        writer.writerows(manifest)

    licenses = destination / 'third_party' / 'licenses'
    shutil.copytree(PREFIX / 'share' / 'licenses', licenses)
    common = pathlib.Path('C:/msys64/usr/share/licenses/common')
    if common.is_dir():
        shutil.copytree(common, licenses / 'common')
    # Several runtime DLLs were reused from the already-installed MinGW toolchain.
    existing_licenses = pathlib.Path('C:/msys64/mingw64/share/licenses')
    for name in ['gcc-libs', 'libwinpthread', 'winpthreads', 'bzip2', 'expat',
                 'gettext-runtime', 'libffi', 'libiconv', 'openssl', 'pcre2', 'xz',
                 'zlib', 'zstd']:
        source = existing_licenses / name
        if source.is_dir():
            shutil.copytree(source, licenses / 'reused_mingw_runtime' / name)
    copy(ROOT / '.deps' / 'imgui-1.91.9b' / 'LICENSE.txt', licenses / 'Dear_ImGui_MIT.txt')
    packages, providers = package_metadata()
    dependencies = []
    for package in sorted({providers[k] for k in dlls if k in providers}):
        info = packages.get(package, {})
        short = package.removeprefix('mingw-w64-x86_64-')
        dependencies.append(dict(package=package, version=info.get('VERSION', []),
            licences=info.get('LICENSE', []), upstream=info.get('URL', []),
            official_binary='https://repo.msys2.org/mingw/mingw64/' +
                            ''.join(info.get('FILENAME', [])),
            binary_sha256=info.get('SHA256SUM', []),
            build_recipe='https://github.com/msys2/MINGW-packages/tree/master/mingw-w64-' + short))
    (destination / 'third_party' / 'DEPENDENCY_SOURCES.json').write_text(
        json.dumps(dependencies, indent=2))
    (destination / 'third_party' / 'DATASET_NOTICE.txt').write_text(
        'KolektorSDD2: images provided and annotated by Kolektor Group; dataset distributed '
        'by ViCoS Lab.\n'
        'Authors: Jakob Bozic, Domen Tabernik, Danijel Skocaj.\n'
        'Source: https://www.vicos.si/resources/kolektorsdd2/\n'
        'Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.\n'
        'https://creativecommons.org/licenses/by-nc-sa/4.0/\n'
        'This trial preserves original sample image and mask bytes.\n'
        'Citation: Mixed supervision for surface-defect detection: from weakly to fully '
        'supervised learning, Computers in Industry, 2021.\n'
        'https://prints.vicos.si/publications/385\n')
    start = r'''@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0bin;%SystemRoot%\System32;%SystemRoot%"
"%~dp0bin\inspect.exe" view --image "data\KolektorSDD2\test\20042.png"
if errorlevel 1 (
  echo Application failed. Please send the displayed error message.
  pause
)
'''
    checks = r'''@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0bin;%SystemRoot%\System32;%SystemRoot%"
if not exist "checks" mkdir "checks"
copy /y "outputs\demo_sequence.avi" "checks\demo_sequence.avi" >nul
>"checks\check_log.txt" echo Portable project checks - %DATE% %TIME%
echo Running geometry and feature tests...
"bin\core_tests.exe" >>"checks\check_log.txt" 2>&1
if errorlevel 1 goto failed
echo Running real image and video synchronization checks...
"bin\inspect.exe" verify --output checks >>"checks\check_log.txt" 2>&1
if errorlevel 1 goto failed
echo Opening the automated OpenGL test window...
"bin\inspect.exe" view --smoke --output checks --image "data\KolektorSDD2\test\20042.png" >>"checks\check_log.txt" 2>&1
if errorlevel 1 goto failed
echo PASS - all automated checks completed.
echo PASS - all automated checks completed.>>"checks\check_log.txt"
echo Results, genuine screenshots and logs are in the checks folder.
if /i "%~1"=="--no-pause" exit /b 0
pause
exit /b 0
:failed
echo A check failed. Please send checks\check_log.txt.
type "checks\check_log.txt"
if /i "%~1"=="--no-pause" exit /b 1
pause
exit /b 1
'''
    for name, content in [('START_APP.cmd', start), ('RUN_CHECKS.cmd', checks)]:
        (destination / name).write_bytes(content.replace('\n', '\r\n').encode('ascii'))
    normal = len(manifest) - sum(int(r['label']) for r in manifest)
    quickstart = f'''# Portable Windows x64 trial - pehle yeh padho

1. ZIP par **Extract All** karo. App ZIP ke andar se mat chalao.
2. **START_APP.cmd** double-click karo. Python, compiler aur training ki zaroorat nahi.
3. **RUN_CHECKS.cmd** double-click karo. Geometry/features, real video synchronization aur
   actual OpenGL rendering checks chalenge. Screenshots aur log **checks/** mein milenge.
4. Apni image/video try karne ke liye app mein **Open image / Open video** use karo.
   Sample video: **outputs/demo_sequence.avi**. Wheel zoom, right-drag pan, rotation slider,
   region picking, ROI drawing/clipping aur ground-truth toggle try karo.

Target: **Windows 10/11 64-bit, graphics driver supporting OpenGL 3.3**.
This is the same real trained pipeline; it does not retrain on launch.

Included: executable + required DLLs, saved model/configuration, full C++ project source,
{len(manifest)} real held-out sample image/mask pairs ({normal} normal / {len(manifest)-normal} defective),
24-frame still-image video demonstration, original measured results and screenshots.
**DEMO_SAMPLES.csv** contains labels and reference predictions, including real mistakes.
The full 853 MB dataset ZIP is omitted. The included subset is for trial and debugging;
it cannot reproduce full test metrics by itself. For full training/evaluation, first download
the complete dataset using **tools/prepare_dataset.py**, then follow the build commands below.

Feedback mein yeh bhejna: app khula ya nahi; image/video, zoom/pan/rotation, ROI/picking mein
koi bug; suspicious prediction ka image ID; aur **checks/check_log.txt**.
Model perfect nahi hai: the official full test had 48 false alarms and 22 misses. Candidate
localization has mean Dice 0.296. The video is a dataset-stills demonstration, not independent
video evaluation. The measurements below are from the original machine, not your machine.

Original reference results are in **outputs/**; this computer's fresh checks go into **checks/**.
Dependency notices/licences and source/build-recipe links are in **third_party/**.

---

'''
    (destination / 'README.md').write_text(
        quickstart + (ROOT / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
    entries = []
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            entries.append(dict(path=path.relative_to(destination).as_posix(),
                                bytes=path.stat().st_size, sha256=sha256(path)))
    description = dict(project='AI-Assisted Surface Defect Detection and Interactive Visualization',
        target='Windows x64, OpenGL 3.3', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dll_count=len(dlls), demo_sample_pairs=len(samples), full_dataset_included=False,
        system_dlls_not_bundled=windows, files=entries)
    (destination / 'PACKAGE_MANIFEST.json').write_text(json.dumps(description, indent=2))
    archive = share / (ARCHIVE_ROOT + '.zip')
    if archive.exists():
        archive = share / (ARCHIVE_ROOT + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                z.write(path, ARCHIVE_ROOT + '/' + path.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('Packaged ZIP CRC error: ' + bad)
    result = dict(zip=str(archive), bytes=archive.stat().st_size, sha256=sha256(archive),
                  dlls=len(dlls), sample_pairs=len(samples), staging=str(staging))
    remove_own_staging(staging)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
