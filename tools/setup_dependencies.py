"""Project-local, coherent MSYS2 MinGW64 binary dependencies from official sources.
Only runtime imports of the used OpenCV modules are resolved, avoiding Qt/VTK.
"""
import compression.zstd, io, json, pathlib, re, subprocess, tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS = ROOT / '.deps'
CACHE = DEPS / 'cache'
PREFIX = DEPS / 'mingw64'
REPO = 'https://repo.msys2.org/mingw/mingw64/'

def download(url, path):
    if path.exists(): return
    partial = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl.exe', '--fail', '--location', '--retry', '3', '--connect-timeout', '25',
        '--max-time', '600', '-o', str(partial), url], check=True, stdout=subprocess.DEVNULL)
    partial.replace(path)

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    download(REPO + 'mingw64.db', CACHE / 'mingw64.db')
    packages = {}
    with tarfile.open(CACHE / 'mingw64.db', 'r:*') as t:
        for m in t.getmembers():
            if not m.name.endswith('/desc'): continue
            s = t.extractfile(m).read().decode()
            fields = dict(re.findall(r'%([^%]+)%\n(.*?)(?=\n%|\Z)', s, re.S))
            name = fields['NAME'].strip()
            packages[name] = {k:v.strip().splitlines() for k,v in fields.items()}
    download(REPO + 'mingw64.files', CACHE / 'mingw64.files')
    dll_packages = {}
    with tarfile.open(CACHE / 'mingw64.files', 'r:*') as t:
        for m in t:
            if not m.name.endswith('/files'): continue
            package_dir = m.name.split('/')[0]
            package_name = re.sub(r'-[^-]+-[^-]+$', '', package_dir)
            if package_name not in packages: continue
            for filename in t.extractfile(m).read().decode().splitlines():
                if filename.startswith('mingw64/bin/') and filename.lower().endswith('.dll'):
                    dll_packages[filename.split('/')[-1].lower()] = package_name
    installed = set()
    def install(short):
        name = short if short.startswith('mingw-w64-') else 'mingw-w64-x86_64-' + short
        if name in installed: return
        info = packages[name]
        filename = info['FILENAME'][0]
        archive = CACHE / filename
        download(REPO + filename, archive)
        import hashlib
        if hashlib.sha256(archive.read_bytes()).hexdigest() != info['SHA256SUM'][0]:
            raise RuntimeError('Checksum mismatch: ' + filename)
        # Package names and paths are constrained to the official mingw64 prefix.
        with archive.open('rb') as f:
            with compression.zstd.ZstdFile(f) as decompressed:
                with tarfile.open(fileobj=decompressed, mode='r|') as t:
                    for m in t:
                        p = pathlib.PurePosixPath(m.name)
                        if not p.parts or p.parts[0] != 'mingw64': continue
                        if '..' in p.parts or m.issym() or m.islnk(): continue
                        out = DEPS / str(p)
                        if m.isdir(): out.mkdir(parents=True, exist_ok=True)
                        elif m.isfile():
                            out.parent.mkdir(parents=True, exist_ok=True)
                            with t.extractfile(m) as source, out.open('wb') as dest:
                                import shutil
                                shutil.copyfileobj(source, dest)
        installed.add(name)
        print('Installed locally: ' + filename, flush=True)
    for name in ['cmake', 'ninja', 'opencv', 'glfw', 'glew']: install(name)
    known = {'libjpeg-8.dll':'libjpeg-turbo','libpng16-16.dll':'libpng','libtiff-6.dll':'libtiff',
       'libwebp-7.dll':'libwebp','libwebpdemux-2.dll':'libwebp','libwebpmux-3.dll':'libwebp',
       'libopenjp2-7.dll':'openjpeg2', 'libjasper-7.dll':'jasper','liblzma-5.dll':'xz',
       'libbz2-1.dll':'bzip2','zlib1.dll':'zlib','libdeflate.dll':'libdeflate','libzstd.dll':'zstd',
       'libjbig-0.dll':'jbigkit','libLerc.dll':'lerc','libwinpthread-1.dll':'winpthreads',
       'libgcc_s_seh-1.dll':'gcc-libs','libstdc++-6.dll':'gcc-libs', 'libintl-8.dll':'gettext-runtime',
       'libiconv-2.dll':'libiconv','libdouble-conversion.dll':'double-conversion',
       'libImath-3_2.dll':'imath','libIlmThread-3_4.dll':'openexr','libIex-3_4.dll':'openexr',
       'libOpenEXR-3_4.dll':'openexr','libOpenEXRCore-3_4.dll':'openexr',
       'libavif-16.dll':'libavif','libaom.dll':'aom','libdav1d-7.dll':'dav1d',
       'libsharpyuv-0.dll':'libwebp','libbrotlicommon.dll':'brotli','libbrotlidec.dll':'brotli',
       'libarchive-13.dll':'libarchive','libcurl-4.dll':'curl','libssl-3-x64.dll':'openssl',
       'libcrypto-3-x64.dll':'openssl','libexpat-1.dll':'expat','libuv-1.dll':'libuv',
       'libnghttp2-14.dll':'nghttp2','libnghttp3-9.dll':'nghttp3','libngtcp2-16.dll':'ngtcp2',
       'libngtcp2_crypto_ossl-0.dll':'ngtcp2','libssh2-1.dll':'libssh2','libpsl-5.dll':'libpsl',
       'libidn2-0.dll':'libidn2','libunistring-5.dll':'libunistring', 'libhwy.dll':'highway',
       'libjxl.dll':'libjxl','libjxl_threads.dll':'libjxl','libjxl_cms.dll':'libjxl',
       'libOpenEXR-3_3.dll':'openexr','libOpenEXRCore-3_3.dll':'openexr',
       'libIlmThread-3_3.dll':'openexr','libIex-3_3.dll':'openexr','libImath-3_1.dll':'imath'}
    queue = [*PREFIX.glob('bin/libopencv_core*.dll'), *PREFIX.glob('bin/libopencv_imgproc*.dll'),
             *PREFIX.glob('bin/libopencv_imgcodecs*.dll'), *PREFIX.glob('bin/libopencv_ml*.dll'),
             *PREFIX.glob('bin/libopencv_videoio*.dll'), *PREFIX.glob('bin/libopencv_geometry*.dll'), *PREFIX.glob('bin/glfw3.dll'),
             *PREFIX.glob('bin/glew32.dll'), PREFIX/'bin/cmake.exe', PREFIX/'bin/ninja.exe']
    visited = set()
    while queue:
        dll = queue.pop()
        if dll.name.lower() in visited: continue
        visited.add(dll.name.lower())
        output = subprocess.check_output(['C:/msys64/mingw64/bin/objdump.exe','-p',str(dll)], text=True)
        for dep in re.findall(r'DLL Name: (.+)', output):
            dep = dep.strip()
            local = PREFIX/'bin'/dep
            if not local.exists():
                existing = pathlib.Path('C:/msys64/mingw64/bin')/dep
                if existing.exists():
                    import shutil
                    shutil.copy2(existing,local)
                elif dep.lower() in dll_packages: install(dll_packages[dep.lower()])
                elif dep in known: install(known[dep])
                elif re.match(r'lib(.*?)-\d.*\.dll', dep):
                    stem = re.match(r'lib(.*?)-\d.*\.dll',dep).group(1)
                    choices = [k for k in packages if k.endswith('-'+stem) or k.endswith('-lib'+stem)]
                    if len(choices)==1: install(choices[0])
                    else: raise RuntimeError('Unmapped runtime DLL: ' + dep)
                elif not (pathlib.Path('C:/Windows/System32')/dep).exists() and not dep.lower().startswith('api-ms-'):
                    raise RuntimeError('Unmapped runtime DLL: ' + dep)
            if local.exists(): queue.append(local)
    download('https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9b.tar.gz',CACHE/'imgui.tar.gz')
    with tarfile.open(CACHE/'imgui.tar.gz') as t:
        t.extractall(DEPS, filter='data')
    (DEPS/'versions.json').write_text(json.dumps(sorted(installed), indent=2))
    print('Dependency setup complete', flush=True)

if __name__ == '__main__': main()
