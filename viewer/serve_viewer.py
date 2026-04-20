#!/usr/bin/env python3
"""JAXTPC Viewer — local file server with HTTP Range support.

Serves production HDF5 files + the viewer frontend. Files are matched
by dataset_name (embedded in filenames as {dataset}_{kind}_{batch}.h5).

Supports both flat directories and seg/inst/sensor subdirectory layouts.

Usage:
    python3 viewer/serve_viewer.py production_run/
    python3 viewer/serve_viewer.py production_run/ --dataset myrun
    python3 viewer/serve_viewer.py production_run/ --port 9000 --open
"""

import os
import re
import sys
import json
import argparse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from glob import glob

KINDS = ('seg', 'inst', 'sensor')
OPTIONAL_KINDS = ('labl', 'optical')


# ── File discovery ──────────────────────────────────────────────

def find_h5_files(prod_dir):
    """Find all *_{kind}_*.h5 files, searching both flat and subdirectory layouts."""
    found = {}  # kind -> [abspath, ...]
    for kind in KINDS + OPTIONAL_KINDS:
        files = []
        # Subdirectory layout: prod_dir/seg/*.h5
        sub = os.path.join(prod_dir, kind)
        if os.path.isdir(sub):
            files += glob(os.path.join(sub, f'*_{kind}_*.h5'))
        # Flat layout: prod_dir/*_seg_*.h5
        files += glob(os.path.join(prod_dir, f'*_{kind}_*.h5'))
        # Deduplicate and filter
        files = sorted(set(os.path.abspath(f) for f in files))
        files = [f for f in files if '_lzf' not in f]
        if files:
            found[kind] = files
    return found


def extract_dataset(filename):
    """Extract dataset name from '{dataset}_{kind}_{batch}.h5'."""
    base = os.path.basename(filename)
    for kind in KINDS + OPTIONAL_KINDS:
        m = re.match(rf'^(.+)_{kind}_\d+\.h5$', base)
        if m:
            return m.group(1)
    return None


def discover_datasets(prod_dir):
    """Return {dataset_name: {kind: filepath, ...}} for all complete datasets."""
    all_files = find_h5_files(prod_dir)
    # Group by dataset name
    datasets = {}
    for kind, files in all_files.items():
        for f in files:
            ds = extract_dataset(f)
            if ds:
                datasets.setdefault(ds, {})[kind] = f
    return datasets


def select_dataset(prod_dir, requested=None):
    """Select a dataset and return its manifest dict."""
    datasets = discover_datasets(prod_dir)

    if not datasets:
        sys.exit(f"Error: no HDF5 files matching *_{{seg,inst,sensor}}_*.h5 found in {prod_dir}")

    if requested:
        if requested not in datasets:
            available = ', '.join(sorted(datasets.keys()))
            sys.exit(f"Error: dataset '{requested}' not found. Available: {available}")
        ds = requested
    elif len(datasets) == 1:
        ds = next(iter(datasets))
    else:
        # Multiple datasets — pick the one with all three required kinds
        complete = {k: v for k, v in datasets.items() if all(kind in v for kind in KINDS)}
        if len(complete) == 1:
            ds = next(iter(complete))
        else:
            candidates = complete or datasets
            print(f"Multiple datasets found in {prod_dir}:")
            for name, kinds in sorted(candidates.items()):
                labels = ', '.join(sorted(kinds.keys()))
                print(f"  {name}  ({labels})")
            sys.exit("Use --dataset <name> to select one.")

    info = datasets[ds]
    missing = [k for k in KINDS if k not in info]
    if missing:
        sys.exit(f"Error: dataset '{ds}' missing {', '.join(missing)} files")

    return ds, info


def build_manifest(prod_dir, file_map):
    """Convert absolute paths to relative paths for the manifest."""
    manifest = {}
    for kind, abspath in file_map.items():
        manifest[kind] = os.path.relpath(abspath, prod_dir)
    return manifest


# ── HTTP server ─────────────────────────────────────────────────

class RangeHandler(SimpleHTTPRequestHandler):
    base_dir = None
    project_dir = None
    manifest = None

    STATIC_FILES = {
        '/viewer.js':      ('viewer.js', 'application/javascript'),
        '/viewer.css':     ('viewer.css', 'text/css'),
        '/shaders.js':     ('shaders.js', 'application/javascript'),
        '/colormaps.js':   ('colormaps.js', 'application/javascript'),
        '/h5_worker.js':   ('h5_worker.js', 'application/javascript'),
    }

    def do_HEAD(self):
        path = self._resolve_path()
        if not path:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', os.path.getsize(path))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Expose-Headers',
                         'Content-Length, Content-Range, Accept-Ranges')
        self.end_headers()

    def do_GET(self):
        # Viewer HTML
        if self.path == '/' or self.path.split('?')[0] == '/':
            self._send_file(os.path.join(self.project_dir, 'index.html'),
                            'text/html')
            return
        # Static viewer files
        if self.path in self.STATIC_FILES:
            rel, ct = self.STATIC_FILES[self.path]
            self._send_file(os.path.join(self.project_dir, rel), ct)
            return
        # Manifest
        if self.path == '/manifest.json':
            body = json.dumps(self.manifest).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
            return
        # HDF5 files with Range support
        path = self._resolve_path()
        if not path:
            self.send_error(404)
            return
        file_size = os.path.getsize(path)
        rng = self.headers.get('Range')
        if rng:
            try:
                spec = rng.replace('bytes=', '')
                parts = spec.split('-')
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
            except (ValueError, IndexError):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            self.send_header('Content-Length', length)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Expose-Headers',
                             'Content-Length, Content-Range, Accept-Ranges')
            self.end_headers()
            with open(path, 'rb') as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self._send_file(path, 'application/octet-stream')

    def _resolve_path(self):
        """Map URL path to a real file under base_dir."""
        clean = self.path.split('?')[0].lstrip('/')
        if not clean:
            return None
        real = os.path.normpath(os.path.join(self.base_dir, clean))
        if not real.startswith(self.base_dir) or not os.path.isfile(real):
            return None
        return real

    def _send_file(self, path, content_type):
        if not os.path.exists(path):
            self.send_error(404)
            return
        data = open(path, 'rb').read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if 'Range' not in (self.headers.get('Range') or ''):
            super().log_message(fmt, *args)


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='JAXTPC Viewer — serve production data for browser visualization')
    parser.add_argument('data_dir', help='Directory containing production HDF5 files')
    parser.add_argument('--dataset', '-d', help='Dataset name (auto-detected if only one)')
    parser.add_argument('--port', '-p', type=int, default=8765)
    parser.add_argument('--open', '-o', action='store_true',
                        help='Open browser automatically')
    args = parser.parse_args()

    prod_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(prod_dir):
        sys.exit(f"Error: {prod_dir} is not a directory")
    project_dir = os.path.dirname(os.path.abspath(__file__))

    dataset_name, file_map = select_dataset(prod_dir, args.dataset)
    manifest = build_manifest(prod_dir, file_map)

    print(f"=== JAXTPC Viewer ===")
    print(f"Dataset: {dataset_name}")
    for k, v in sorted(manifest.items()):
        sz = os.path.getsize(os.path.join(prod_dir, v))
        label = f'{sz/1e9:.2f} GB' if sz > 1e8 else f'{sz/1e6:.1f} MB'
        print(f"  {k:8s}  {v}  ({label})")

    RangeHandler.base_dir = prod_dir
    RangeHandler.project_dir = project_dir
    RangeHandler.manifest = manifest

    url = f'http://127.0.0.1:{args.port}/'

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedServer(('127.0.0.1', args.port), RangeHandler)
    print(f"\n{url}")
    print("Ctrl+C to stop\n")

    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
