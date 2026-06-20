#!/usr/bin/env python3
"""
FastAPI-based video streaming server with optimized performance.
Usage: python fastapi_video_server.py --port 9009
"""
import os
import argparse
import subprocess
import shutil
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="ANPR Video Player")
WEB_CACHE_DIR = Path(".web_cache")
SUPPORTED_CODECS = {"h264", "hevc", "vp8", "vp9", "av1"}


def remove_tree(target: Path):
    def handle_remove_error(func, path, exc_info):
        try:
            os.chmod(path, 0o700)
            func(path)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(target, onerror=handle_remove_error)


def get_video_list():
    """Scan directory for MP4 files."""
    current_dir = Path.cwd()
    videos = []
    for video_file in current_dir.rglob('*.mp4'):
        rel_path = video_file.relative_to(current_dir)
        videos.append({
            'name': video_file.name,
            'path': str(rel_path),
            'size': video_file.stat().st_size,
            'folder': str(rel_path.parent)
        })
    videos.sort(key=lambda x: (x['folder'], x['name']))
    return videos


def detect_video_codec(file_path: Path) -> str:
    """Return video codec name using ffprobe, or empty string on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def ensure_browser_playable(video_path: Path, rel_path: str) -> Path:
    """
    If codec is unsupported in browser, transcode once to H.264 and cache it.
    Returns the path to stream.
    """
    codec = detect_video_codec(video_path)
    if codec in SUPPORTED_CODECS:
        return video_path

    target_rel = Path(rel_path).with_suffix(".h264.mp4")
    target_path = WEB_CACHE_DIR / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target_path


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def ranged_response(file_path: Path, request: Request):
    """Stream video file with HTTP range support."""
    file_size = file_path.stat().st_size
    range_header = request.headers.get('range')

    if range_header:
        # Parse range header: bytes=start-end
        range_match = range_header.replace('bytes=', '').split('-')
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        chunk_size = end - start + 1

        def file_iterator():
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(chunk_size),
            'Content-Type': 'video/mp4',
        }
        return StreamingResponse(file_iterator(), status_code=206, headers=headers)

    # No range request - stream entire file
    def file_iterator():
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                yield chunk

    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(file_size),
        'Content-Type': 'video/mp4',
    }
    return StreamingResponse(file_iterator(), headers=headers)


def resolve_folder_path(folder_path: str) -> Path:
    """Resolve and validate folder path under current working directory."""
    decoded = urllib.parse.unquote(folder_path).strip()
    if not decoded or decoded == ".":
        raise HTTPException(status_code=400, detail="Refusing to delete root folder")
    if ".." in Path(decoded).parts:
        raise HTTPException(status_code=400, detail="Invalid folder path")
    base = Path.cwd().resolve()
    target = (base / decoded).resolve()
    if base not in target.parents:
        raise HTTPException(status_code=400, detail="Folder must be inside working directory")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    return target


@app.get("/", response_class=HTMLResponse)
async def index():
    """Main page with video grid."""
    videos = get_video_list()

    video_cards = '\n'.join([f'''
    <div class="video-card">
        <div class="video-container">
            <div class="play-overlay" onclick="loadVideo(this.parentElement)">
                <div class="play-btn"><div class="play-icon"></div></div>
            </div>
            <video controls preload="none" data-src="/video/{urllib.parse.quote(v['path'])}"></video>
        </div>
        <div class="video-info">
            <div class="video-name">{v['name']}</div>
            <div class="video-meta">Size: {format_size(v['size'])}</div>
            <div class="folder-row">
                <div class="folder-badge">{v['folder'] if v['folder'] != '.' else 'root'}</div>
                <button class="delete-btn" {'disabled' if v['folder'] == '.' else ''} onclick="deleteFolder('{urllib.parse.quote(v['folder'])}')">
                    Delete Folder
                </button>
            </div>
        </div>
    </div>''' for v in videos]) if videos else '<div class="no-videos">No MP4 videos found.</div>'

    html = f"""<!DOCTYPE html>
<html><head><title>ANPR Video Player</title><meta charset="UTF-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ font-size: 2rem; margin-bottom: 10px; color: #60a5fa; }}
.info {{ color: #94a3b8; margin-bottom: 30px; font-size: 0.9rem; }}
.video-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 20px; }}
.video-card {{ background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; transition: all 0.2s; }}
.video-card:hover {{ border-color: #60a5fa; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(96,165,250,0.2); }}
.video-container {{ position: relative; width: 100%; padding-top: 56.25%; background: #000; }}
.play-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.7); cursor: pointer; transition: all 0.3s; z-index: 10; }}
.play-overlay:hover {{ background: rgba(0,0,0,0.85); }}
.play-btn {{ width: 80px; height: 80px; border-radius: 50%; background: #60a5fa; display: flex; align-items: center; justify-content: center; transition: transform 0.2s; }}
.play-overlay:hover .play-btn {{ transform: scale(1.15); background: #3b82f6; }}
.play-icon {{ width: 0; height: 0; border-left: 25px solid white; border-top: 15px solid transparent; border-bottom: 15px solid transparent; margin-left: 8px; }}
video {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; display: none; }}
video.loaded {{ display: block; }}
.video-info {{ padding: 15px; }}
.video-name {{ font-weight: 600; color: #f1f5f9; margin-bottom: 5px; word-break: break-word; }}
.video-meta {{ font-size: 0.85rem; color: #94a3b8; }}
.folder-row {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 5px; }}
.folder-badge {{ display: inline-block; background: #334155; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; margin-top: 5px; }}
.delete-btn {{ border: 1px solid #ef4444; color: #fecaca; background: #7f1d1d; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; cursor: pointer; }}
.delete-btn:hover {{ background: #991b1b; }}
.delete-btn:disabled {{ opacity: 0.5; cursor: not-allowed; border-color: #475569; background: #334155; color: #94a3b8; }}
.no-videos {{ text-align: center; padding: 60px 20px; color: #64748b; font-size: 1.2rem; }}
.loading {{ color: #60a5fa; text-align: center; padding: 10px; font-size: 14px; }}
</style></head><body>
<div class="container">
    <h1>🎥 ANPR Video Player</h1>
    <div class="info">Found <strong>{len(videos)}</strong> video(s) | Click to stream | Press <strong>Space</strong> to play/pause active video</div>
    <div class="video-grid">{video_cards}</div>
</div>
<script>
let activeVideo = null;

function loadVideo(container) {{
    const overlay = container.querySelector('.play-overlay');
    const video = container.querySelector('video');
    const videoSrc = video.dataset.src;
    activeVideo = video;

    if (!video.src) {{
        overlay.innerHTML = '<div class="loading">⏳ Loading...</div>';
        video.src = videoSrc;
        video.preload = 'metadata';

        video.onloadedmetadata = function() {{
            console.log('✓ Video ready:', videoSrc);
            video.classList.add('loaded');
            overlay.style.display = 'none';
            video.play().catch(e => {{
                console.error('Play error:', e);
                overlay.style.display = 'flex';
                overlay.innerHTML = '<div style="color: #ef4444;">▶ Click to play</div>';
            }});
        }};

        video.onerror = function(e) {{
            console.error('Video error:', video.error);
            const errors = ['', 'ABORTED', 'NETWORK', 'DECODE', 'NOT_SUPPORTED'];
            const msg = video.error ? errors[video.error.code] : 'UNKNOWN';
            overlay.innerHTML = '<div style="color: #ef4444; font-size: 12px;">Error: ' + msg + '</div>';
        }};
    }} else {{
        overlay.style.display = 'none';
        video.classList.add('loaded');
        video.play();
    }}

    video.addEventListener('click', () => {{
        activeVideo = video;
    }});
}}

async function deleteFolder(encodedFolder) {{
    const folder = decodeURIComponent(encodedFolder);
    if (!folder || folder === '.') {{
        alert('Root folder cannot be deleted.');
        return;
    }}
    const confirmed = confirm(`Delete folder "${{folder}}" and all videos?`);
    if (!confirmed) return;

    try {{
        const resp = await fetch(`/folder/${{encodedFolder}}`, {{ method: 'DELETE' }});
        const data = await resp.json();
        if (!resp.ok) {{
            throw new Error(data.detail || 'Delete failed');
        }}
        alert(data.message || 'Folder deleted');
        location.reload();
    }} catch (err) {{
        console.error('Delete folder error:', err);
        alert(`Failed to delete folder: ${{err.message}}`);
    }}
}}

document.addEventListener('keydown', (event) => {{
    const tag = (event.target && event.target.tagName) ? event.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea') return;
    if (event.code !== 'Space') return;
    if (!activeVideo) return;
    event.preventDefault();
    if (activeVideo.paused) {{
        activeVideo.play().catch(err => console.error('Play error:', err));
    }} else {{
        activeVideo.pause();
    }}
}});
</script>
</body></html>"""
    return HTMLResponse(content=html)


@app.get("/video/{file_path:path}")
async def stream_video(file_path: str, request: Request):
    """Stream video file with range support."""
    decoded = urllib.parse.unquote(file_path)
    video_path = Path.cwd() / decoded

    if not video_path.exists() or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        stream_path = ensure_browser_playable(video_path, decoded)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Video transcoding failed")

    return ranged_response(stream_path, request)


@app.delete("/folder/{folder_path:path}")
async def delete_folder(folder_path: str):
    """Delete a folder and all contained files under current working directory."""
    target = resolve_folder_path(folder_path)
    rel_target = target.relative_to(Path.cwd().resolve())

    cache_target = (WEB_CACHE_DIR / rel_target).resolve()
    try:
        remove_tree(target)
        if cache_target.exists() and cache_target.is_dir():
            remove_tree(cache_target)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=f"Folder could not be deleted: {exc}")

    return {
        "status": "ok",
        "message": f"Deleted folder: {rel_target}",
    }


def main():
    parser = argparse.ArgumentParser(description='FastAPI video streaming server')
    parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host (default: 0.0.0.0)')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"🚀 FastAPI Video Streaming Server")
    print(f"{'=' * 60}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Local:  http://localhost:{args.port}")
    print(f"Path:   {Path.cwd()}")
    print(f"\nFeatures: HTTP Range Requests, Lazy Loading, Chunked Streaming")
    print(f"Press Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
