"""Download, integrity-check and safely preserve the official KolektorSDD2 archive."""
import argparse, concurrent.futures, datetime, hashlib, json, pathlib, re, shutil, subprocess, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = 'https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--connections', type=int, default=4, choices=range(1,5))
    args = parser.parse_args()
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    archive = data / 'KolektorSDD2.zip'
    partial = archive.with_suffix('.zip.part')
    if not archive.exists():
        if args.connections == 1:
            subprocess.run(['curl.exe', '--fail', '--location', '--retry', '3',
                '--retry-delay', '3', '--connect-timeout', '25', '--max-time', '1800',
                '--continue-at', '-', '--output', str(partial), URL], check=True)
        else:
            headers = subprocess.check_output(['curl.exe','--fail','--silent','--show-error',
                '--location','--head','--connect-timeout','25','--max-time','45',URL],text=True)
            if 'application/zip' not in headers.lower(): raise RuntimeError('Dataset response is not ZIP content')
            length = int(re.findall(r'(?im)^content-length:\s*(\d+)',headers)[-1])
            prefix = partial.stat().st_size if partial.exists() else 0
            if prefix > length: raise RuntimeError('Partial file larger than response')
            ranges=[]
            step=(length-prefix+args.connections-1)//args.connections
            for start in range(prefix,length,max(1,step)):
                end=min(length-1,start+step-1)
                ranges.append((start,end,data/f'KolektorSDD2.range_{start}_{end}.part'))
            def fetch(job):
                start,end,path=job
                if path.exists() and path.stat().st_size==end-start+1: return
                print(f'Downloading bytes {start}-{end}',flush=True)
                subprocess.run(['curl.exe','--fail','--silent','--show-error','--location',
                    '--retry','3','--retry-delay','3','--connect-timeout','25','--max-time','1800',
                    '--range',f'{start}-{end}','--output',str(path),URL],check=True)
                if path.stat().st_size!=end-start+1: raise RuntimeError('Range response size mismatch')
                print(f'Completed bytes {start}-{end}',flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.connections) as pool:
                list(pool.map(fetch,ranges))
            with partial.open('ab') as dest:
                for _,_,path in ranges:
                    with path.open('rb') as source: shutil.copyfileobj(source,dest,1024*1024)
            for _,_,path in ranges: path.unlink()
            if partial.stat().st_size != length: raise RuntimeError('Assembled response size mismatch')
        if not zipfile.is_zipfile(partial):
            raise RuntimeError('Response is not a ZIP; retained partial for diagnosis')
        with zipfile.ZipFile(partial) as z:
            bad = z.testzip()
            if bad: raise RuntimeError(f'ZIP CRC failed: {bad}')
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad: raise RuntimeError(f'ZIP CRC failed: {bad}')
        destination = (data / 'KolektorSDD2').resolve()
        for entry in z.infolist():
            p = pathlib.PurePosixPath(entry.filename.replace('\\', '/'))
            target = (destination / str(p)).resolve()
            if p.is_absolute() or not target.is_relative_to(destination) or ':' in str(p):
                raise RuntimeError(f'Unsafe archive entry: {entry.filename}')
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError(f'Symlink rejected: {entry.filename}')
        print(f'Validated {len(z.infolist())} archive entries, {archive.stat().st_size:,} bytes', flush=True)
        if not args.inspect_only:
            destination.mkdir(exist_ok=True)
            for entry in z.infolist():
                target = destination / entry.filename
                if entry.is_dir(): target.mkdir(parents=True, exist_ok=True)
                elif not target.exists() or target.stat().st_size != entry.file_size:
                    z.extract(entry, destination)
        auxiliary = [e.filename for e in z.infolist() if not e.is_dir() and
            (not e.filename.lower().endswith('.png') or '(copy)' in e.filename.lower())]
    digest = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    metadata = dict(source_url=URL, official_page='https://www.vicos.si/resources/kolektorsdd2/',
        download_date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        bytes=archive.stat().st_size, sha256=digest.hexdigest(), zip_crc_verified=True,
        auxiliary_entries=auxiliary)
    previous = data / 'source.json'
    if previous.exists(): metadata['download_date_utc'] = json.loads(previous.read_text())['download_date_utc']
    previous.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

if __name__ == '__main__': main()
