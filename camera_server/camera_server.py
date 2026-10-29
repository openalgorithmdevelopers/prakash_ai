#!/usr/bin/env python3
"""
============================================================================
Raspberry Pi Zero 2 W Camera Server (picamera2 + aiohttp)
============================================================================
DESCRIPTION:
  Hosts the Raspberry Pi camera over HTTP so that another Raspberry Pi,
  a Jetson or a laptop can fetch images, watch live video and request timed
  recordings. A single Python process owns the camera and serves all
  endpoints. Video uses the Pi's hardware H.264 encoder through picamera2,
  and one encoder is shared by every client watching the same tier.

  Endpoints:
    GET /                              HTML home page with full usage docs
    GET /options                       JSON list of image/video/record tiers
    GET /image?q=i1..i5                one JPEG (5 quality tiers)
    GET /video?q=v1..v4                live H.264 stream (4 quality tiers)
    GET /record?q=r1..r4&t=SECONDS     H.264 recording of t seconds chosen
                                       by the client (4 quality tiers,
                                       t limited by MAX_RECORD_S)

FEATURES:
  - Image, video and recording tiers ordered from highest to lowest quality.
  - Tiers larger than the camera sensor are removed automatically.
  - Hardware H.264 encoding with fan-out: many viewers, one encoder.
  - New viewers start at the next keyframe; slow viewers are dropped back
    to the next keyframe instead of stalling the others.
  - Camera starts on demand and stops IDLE_STOP_S seconds after last use.
  - Optional token authentication (Bearer header or ?token= parameter).
  - Home page is public and fills examples with the server's own address.
  - HTTP 409 when the camera is busy with a different profile.
  - Clean shutdown on Ctrl+C / SIGTERM (camera is released).

CONTROLS:
  Ctrl+C in the terminal stops the server. Do not use Ctrl+Z: it only
  suspends the process, which keeps holding the camera.
  Environment variables:
    CAM_TOKEN   access token (empty or unset = no authentication)
    CAM_PORT    listening port (default 8000)

----------------------------------------------------------------------------
VERSION HISTORY
----------------------------------------------------------------------------
 SNO | Version | Date       | Author           | Description
-----+---------+------------+------------------+---------------------------
  1  | 1.0     | 2026-10-26 | Bhupendra Singh  | Initial version: /options,
     |         |            | (Claude)         | /image, /video, /record,
     |         |            |                  | token auth, idle shutdown.
-----+---------+------------+------------------+---------------------------
  2  | 1.1     | 2026-10-26 | Bhupendra Singh  | Added public HTML home
     |         |            | (Claude)         | page with usage examples.
----------------------------------------------------------------------------

SYSTEM PREREQUISITES & PERMISSIONS:
  1. Raspberry Pi Zero 2 W with Raspberry Pi OS Lite (32-bit, Bookworm).
  2. Camera connected and detected:
       rpicam-hello --list-cameras
  3. The user running the server must be in the 'video' group (the default
     for the user created by Raspberry Pi Imager).
  4. No other program may use the camera while the server runs
     (stop rpicam-* tools and any previous copy of this server).

DEPENDENCIES INSTALLATION:
  sudo apt update
  sudo apt install -y --no-install-recommends python3-picamera2 python3-aiohttp

COMPILATION:
  Not required (interpreted Python 3). Optional syntax check:
    python3 -m py_compile server.py

RUN:
  CAM_TOKEN=secret python3 server.py
  Then open http://<pi-ip>:8000/ in a browser for the documentation.
============================================================================
"""
import asyncio
import hmac
import io
import logging
import os

from aiohttp import web
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import Output

PORT = int(os.environ.get('CAM_PORT', '8000'))
TOKEN = os.environ.get('CAM_TOKEN', '')   # empty = no authentication
MAX_RECORD_S = 60     # upper limit for t
IDLE_STOP_S = 8       # stop the camera this long after the last user leaves
STALL_S = 10          # drop a client if no frame arrives for this long
PUBLIC_PATHS = ('/', '/favicon.ico')   # documentation page: no token needed

# id: (width, height, jpeg quality), highest to lowest.
# Tiers larger than the sensor are removed at start-up.
IMAGE_TIERS = {
    'i1': (3280, 2464, 95),
    'i2': (1920, 1080, 90),
    'i3': (1280, 720, 85),
    'i4': (640, 480, 80),
    'i5': (320, 240, 70),
}
# id: (width, height, fps, bitrate in bit/s), highest to lowest
VIDEO_TIERS = {
    'v1': (1920, 1080, 30, 6_000_000),
    'v2': (1280, 720, 30, 3_000_000),
    'v3': (640, 480, 25, 1_000_000),
    'v4': (320, 240, 15, 300_000),
}
RECORD_TIERS = {
    'r1': (1920, 1080, 30, 6_000_000),
    'r2': (1280, 720, 30, 3_000_000),
    'r3': (640, 480, 25, 1_000_000),
    'r4': (320, 240, 15, 300_000),
}

log = logging.getLogger('camserver')


class Client:
    def __init__(self):
        self.q = asyncio.Queue(maxsize=150)
        self.synced = False      # becomes True at the first keyframe


class Broadcaster(Output):
    # Receives encoded H.264 frames from the encoder thread, copies them to every client.
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.clients = set()

    def outputframe(self, frame, keyframe=True, timestamp=None, *args, **kwargs):
        self.loop.call_soon_threadsafe(self._fan, bytes(frame), keyframe)

    def _fan(self, data, keyframe):
        for c in self.clients:
            if not c.synced:
                if not keyframe:
                    continue          # new client: wait for a keyframe so decoding can start
                c.synced = True
            try:
                c.q.put_nowait(data)
            except asyncio.QueueFull:
                while not c.q.empty():   # slow client: drop backlog, resync at next keyframe
                    c.q.get_nowait()
                c.synced = False


class CameraManager:
    def __init__(self, loop):
        self.loop = loop
        self.cam = Picamera2()
        self.bcast = Broadcaster(loop)
        self.lock = asyncio.Lock()
        self.mode = None            # None, ('still', (w, h)) or ('video', profile)
        self.idle = None
        sw, sh = self.cam.sensor_resolution
        self.image_tiers = {k: v for k, v in IMAGE_TIERS.items() if v[0] <= sw and v[1] <= sh}
        log.info('sensor %dx%d, image tiers %s', sw, sh, list(self.image_tiers))

    def run(self, fn, *args):
        return self.loop.run_in_executor(None, fn, *args)

    # ---- blocking helpers: run in a worker thread, always called under self.lock ----
    def _stop(self):
        if self.mode is None:
            return
        if self.mode[0] == 'video':
            self.cam.stop_recording()
        else:
            self.cam.stop()
        self.mode = None

    def _start_video(self, p):
        w, h, fps, bitrate = p
        cfg = self.cam.create_video_configuration(
            main={'size': (w, h), 'format': 'YUV420'},
            controls={'FrameRate': fps})
        self.cam.configure(cfg)
        enc = H264Encoder(bitrate=bitrate, repeat=True, iperiod=fps)
        self.cam.start_recording(enc, self.bcast)
        self.mode = ('video', p)

    def _start_still(self, size):
        self.cam.configure(self.cam.create_still_configuration(main={'size': size}))
        self.cam.start()
        self.mode = ('still', size)

    def _capture(self, quality):
        self.cam.options['quality'] = quality
        buf = io.BytesIO()
        self.cam.capture_file(buf, format='jpeg')
        return buf.getvalue()

    # ---- idle shutdown ----
    def _cancel_idle(self):
        if self.idle:
            self.idle.cancel()
            self.idle = None

    def _schedule_idle(self):
        self._cancel_idle()
        self.idle = self.loop.call_later(
            IDLE_STOP_S, lambda: asyncio.ensure_future(self._idle_stop()))

    async def _idle_stop(self):
        async with self.lock:
            if not self.bcast.clients:
                await self.run(self._stop)
                log.info('camera idle, stopped')

    # ---- API used by the request handlers ----
    async def acquire_video(self, profile, client):
        async with self.lock:
            self._cancel_idle()
            if self.mode != ('video', profile):
                if self.bcast.clients:
                    return 'camera is busy streaming another profile'
                await self.run(self._stop)
                await self.run(self._start_video, profile)
            self.bcast.clients.add(client)
        return None

    def release_video(self, client):
        self.bcast.clients.discard(client)
        if not self.bcast.clients:
            self._schedule_idle()

    async def snapshot(self, size, quality):
        async with self.lock:
            if self.bcast.clients:
                return None
            self._cancel_idle()
            if self.mode != ('still', size):
                await self.run(self._stop)
                await self.run(self._start_still, size)
                await asyncio.sleep(1.0)    # let auto-exposure and white balance settle
            data = await self.run(self._capture, quality)
            self._schedule_idle()
            return data

    async def shutdown(self):
        self._cancel_idle()
        async with self.lock:
            await self.run(self._stop)
            await self.run(self.cam.close)


@web.middleware
async def auth(request, handler):
    if TOKEN and request.path not in PUBLIC_PATHS:
        given = request.headers.get('Authorization', '').removeprefix('Bearer ')
        given = given or request.query.get('token', '')
        if not hmac.compare_digest(given.encode(), TOKEN.encode()):
            raise web.HTTPUnauthorized(text='unauthorized')
    return await handler(request)


async def options(request):
    cm = request.app['cm']
    base = f'{request.scheme}://{request.host}'
    return web.json_response({
        'image': {k: {'width': v[0], 'height': v[1], 'jpeg_quality': v[2],
                      'url': f'{base}/image?q={k}'}
                  for k, v in cm.image_tiers.items()},
        'video': {k: {'width': v[0], 'height': v[1], 'fps': v[2], 'bitrate': v[3],
                      'url': f'{base}/video?q={k}'}
                  for k, v in VIDEO_TIERS.items()},
        'record': {k: {'width': v[0], 'height': v[1], 'fps': v[2], 'bitrate': v[3],
                       'max_seconds': MAX_RECORD_S, 'url': f'{base}/record?q={k}&t=10'}
                   for k, v in RECORD_TIERS.items()},
    })


async def image(request):
    cm = request.app['cm']
    tier = cm.image_tiers.get(request.query.get('q', 'i3'))
    if tier is None:
        raise web.HTTPBadRequest(text='unknown image tier')
    w, h, quality = tier
    data = await cm.snapshot((w, h), quality)
    if data is None:
        return web.Response(status=409, text='camera is busy streaming',
                            headers={'Retry-After': '5'})
    return web.Response(body=data, content_type='image/jpeg',
                        headers={'Cache-Control': 'no-store'})


async def stream_h264(request, profile, duration=None, filename=None):
    cm = request.app['cm']
    loop = asyncio.get_running_loop()
    client = Client()
    err = await cm.acquire_video(profile, client)
    if err:
        return web.Response(status=409, text=err, headers={'Retry-After': '5'})
    headers = {'Content-Type': 'video/h264', 'Cache-Control': 'no-store'}
    if filename:
        headers['Content-Disposition'] = f'attachment; filename={filename}'
    resp = web.StreamResponse(headers=headers)
    try:
        await resp.prepare(request)
        deadline = None
        while True:
            wait = STALL_S if deadline is None else min(STALL_S, deadline - loop.time())
            if wait <= 0:
                break
            try:
                data = await asyncio.wait_for(client.q.get(), wait)
            except asyncio.TimeoutError:
                break
            if duration and deadline is None:
                deadline = loop.time() + duration    # the clock starts at the first keyframe
            await resp.write(data)
        await resp.write_eof()
    except ConnectionResetError:
        pass
    finally:
        cm.release_video(client)
    return resp


async def video(request):
    profile = VIDEO_TIERS.get(request.query.get('q', 'v3'))
    if profile is None:
        raise web.HTTPBadRequest(text='unknown video tier')
    return await stream_h264(request, profile)


async def record(request):
    q = request.query.get('q', 'r3')
    profile = RECORD_TIERS.get(q)
    if profile is None:
        raise web.HTTPBadRequest(text='unknown record tier')
    try:
        t = float(request.query['t'])
    except (KeyError, ValueError):
        raise web.HTTPBadRequest(text='t (seconds) is required')
    if not 0 < t <= MAX_RECORD_S:
        raise web.HTTPBadRequest(text=f't must be above 0 and at most {MAX_RECORD_S}')
    return await stream_h264(request, profile, duration=t,
                             filename=f'rec_{q}_{int(t)}s.h264')


HOME_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi Camera Server</title>
<style>
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6672;--card:#f5f7fa;--line:#d9dee5;--accent:#0b66d0;--code:#eef1f5}
@media (prefers-color-scheme: dark){
:root{--bg:#14171b;--fg:#e6e9ed;--muted:#9aa5b1;--card:#1c2127;--line:#2c333b;--accent:#6aa9ff;--code:#222931}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px 64px}
h1{margin:0 0 4px;font-size:1.8rem}
h2{margin:36px 0 8px;font-size:1.3rem;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{margin:20px 0 6px;font-size:1.05rem}
a{color:var(--accent)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.92em}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;margin:8px 0;line-height:1.45}
pre code{background:none;padding:0}
.muted{color:var(--muted)} .small{font-size:.88rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:16px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:8px 0}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:var(--card)}
input,select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
input{width:100%;max-width:420px}
button{cursor:pointer;background:var(--accent);color:#fff;border-color:var(--accent)}
img#preview{max-width:100%;border-radius:8px;margin-top:10px;display:none}
</style>
</head>
<body>
<main>
<h1>Pi Camera Server</h1>
<p class="muted">Images, live video and timed recordings from a Raspberry Pi camera over plain HTTP.</p>
<p>Base URL: <code>@@BASE@@</code></p>

<div class="card">
<label for="token"><strong>Access token</strong></label>
<span class="muted small"> @@AUTH_NOTE@@</span><br>
<input id="token" type="password" autocomplete="off" placeholder="paste token (optional)">
<p class="muted small" style="margin:8px 0 0">Used only to fill in the links and examples on this page. It is not stored.</p>
</div>

<h2>Endpoints at a glance</h2>
<div class="scroll"><table>
<tr><th>Endpoint</th><th>Returns</th><th>Works in a browser?</th></tr>
<tr><td><code>GET /options</code></td><td>JSON list of all image, video and record tiers (highest to lowest)</td><td>Yes</td></tr>
<tr><td><code>GET /image?q=i1..i5</code></td><td>One JPEG photo</td><td>Yes</td></tr>
<tr><td><code>GET /video?q=v1..v4</code></td><td>Live H.264 stream; stays open until you disconnect</td><td>No, use ffplay, VLC, GStreamer or OpenCV</td></tr>
<tr><td><code>GET /record?q=r1..r4&amp;t=SECONDS</code></td><td>H.264 recording of SECONDS seconds (max @@MAXT@@), then the response ends</td><td>Downloads a file</td></tr>
</table></div>

<h2>GET /options</h2>
<p>Lists every tier this camera offers, each with its resolution and a ready-to-use URL. Ask this first if your client should choose a tier automatically.</p>
<a data-path="/options" href="/options">Open /options</a>
<pre><code>curl 'http://@@HOST@@/options?token=<span class="tok">YOUR_TOKEN</span>'</code></pre>
<pre><code>import requests
opts = requests.get('@@BASE@@/options', params={'token': 'YOUR_TOKEN'}, timeout=5).json()
print(list(opts['image']), list(opts['video']), list(opts['record']))</code></pre>

<h2>GET /image</h2>
<p>Takes one photo and returns it as <code>image/jpeg</code>. Parameter: <code>q</code>, the image tier (default <code>i3</code>).</p>
<div class="scroll"><table>
<tr><th>q</th><th>Resolution</th><th>JPEG quality</th><th>Try it</th></tr>
@@IMG_ROWS@@
</table></div>
<div class="card">
<strong>Try it</strong><br>
<select id="tier">@@IMG_OPTIONS@@</select>
<button type="button" onclick="snap()">Take photo</button>
<span id="status" class="muted small"></span><br>
<img id="preview" alt="camera preview">
</div>
<pre><code>curl -o snap.jpg 'http://@@HOST@@/image?q=i3&amp;token=<span class="tok">YOUR_TOKEN</span>'</code></pre>
<pre><code>import requests
r = requests.get('@@BASE@@/image', params={'q': 'i3', 'token': 'YOUR_TOKEN'}, timeout=20)
r.raise_for_status()
open('snap.jpg', 'wb').write(r.content)</code></pre>
<p class="muted small">The first photo after the camera has been idle takes 1 to 2 seconds longer while exposure settles.</p>

<h2>GET /video</h2>
<p>Streams live video as raw H.264 (Annex-B) with <code>Content-Type: video/h264</code>. It stays open until the client disconnects. Parameter: <code>q</code>, the video tier (default <code>v3</code>). Clients choosing the same tier share one encoder.</p>
<div class="scroll"><table>
<tr><th>q</th><th>Resolution</th><th>FPS</th><th>Bitrate</th></tr>
@@VID_ROWS@@
</table></div>
<h3>Laptop: ffplay</h3>
<pre><code>ffplay -fflags nobuffer -flags low_delay -f h264 'http://@@HOST@@/video?q=v2&amp;token=<span class="tok">YOUR_TOKEN</span>'</code></pre>
<h3>Laptop: VLC</h3>
<p>Media, Open Network Stream, paste the URL. If nothing shows, add the option <code>:demux=h264</code> under &ldquo;Show more options&rdquo;.</p>
<h3>Jetson: GStreamer with hardware decoding</h3>
<pre><code>gst-launch-1.0 souphttpsrc location='http://@@HOST@@/video?q=v2&amp;token=<span class="tok">YOUR_TOKEN</span>' is-live=true ! h264parse ! nvv4l2decoder ! nvvidconv ! autovideosink</code></pre>
<h3>Python: OpenCV</h3>
<pre><code>import cv2
cap = cv2.VideoCapture('@@BASE@@/video?q=v2&amp;token=YOUR_TOKEN', cv2.CAP_FFMPEG)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow('pi camera', frame)
    if cv2.waitKey(1) == 27:   # Esc to quit
        break</code></pre>
<p class="muted small">If OpenCV cannot detect the raw H.264 format on your build, use ffplay or GStreamer instead. A new viewer may wait up to about 1 second for the next keyframe before video starts.</p>

<h2>GET /record</h2>
<p>Records for <code>t</code> seconds and streams the H.264 data as it is produced, then ends. Parameters: <code>q</code>, the record tier (default <code>r3</code>), and <code>t</code>, the length in seconds (required, above 0 and at most @@MAXT@@).</p>
<div class="scroll"><table>
<tr><th>q</th><th>Resolution</th><th>FPS</th><th>Bitrate</th><th>Try it</th></tr>
@@REC_ROWS@@
</table></div>
<pre><code>curl -o rec.h264 'http://@@HOST@@/record?q=r2&amp;t=10&amp;token=<span class="tok">YOUR_TOKEN</span>'</code></pre>
<pre><code>import requests
url = '@@BASE@@/record'
params = {'q': 'r2', 't': 10, 'token': 'YOUR_TOKEN'}
with requests.get(url, params=params, stream=True, timeout=(5, 15)) as r:
    r.raise_for_status()
    with open('rec.h264', 'wb') as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)</code></pre>
<h3>Turn the recording into an MP4 (no re-encoding)</h3>
<pre><code>ffmpeg -framerate 30 -i rec.h264 -c copy rec.mp4</code></pre>
<p class="muted small">The raw stream has no timestamps, so set <code>-framerate</code> to the tier&rsquo;s FPS (30, 25 or 15).</p>

<h2>Authentication</h2>
<p>If the server was started with a token, send it either as a query parameter (<code>?token=...</code>) or as a header:</p>
<pre><code>curl -H 'Authorization: Bearer YOUR_TOKEN' 'http://@@HOST@@/options'</code></pre>
<p>This page needs no token. Traffic is plain HTTP, so use the server only on a trusted network.</p>

<h2>Status codes</h2>
<div class="scroll"><table>
<tr><th>Code</th><th>Meaning</th></tr>
<tr><td>200</td><td>OK</td></tr>
<tr><td>400</td><td>Unknown tier, or <code>t</code> missing or out of range</td></tr>
<tr><td>401</td><td>Missing or wrong token</td></tr>
<tr><td>409</td><td>Camera busy: someone is streaming a different tier, or you asked for an image during a live stream. Retry after a few seconds</td></tr>
</table></div>

<h2>Good to know</h2>
<ul>
<li>Only one video tier runs at a time. Viewers of the same tier share it.</li>
<li><code>/image</code> is unavailable while someone is watching live video.</li>
<li>The camera stops a few seconds after the last user leaves and restarts on the next request.</li>
<li>The Pi uses 2.4 GHz Wi-Fi, so pick a tier your link can sustain. If video stutters, choose a lower tier.</li>
</ul>
</main>
<script>
function tok() { return document.getElementById('token').value.trim(); }
function withTok(path) {
  var t = tok();
  if (!t) return path;
  return path + (path.indexOf('?') >= 0 ? '&' : '?') + 'token=' + encodeURIComponent(t);
}
function refresh() {
  document.querySelectorAll('[data-path]').forEach(function (a) { a.href = withTok(a.dataset.path); });
  document.querySelectorAll('.tok').forEach(function (s) { s.textContent = tok() || 'YOUR_TOKEN'; });
}
function snap() {
  var q = document.getElementById('tier').value;
  var img = document.getElementById('preview');
  var st = document.getElementById('status');
  st.textContent = ' Capturing...';
  img.onload = function () { st.textContent = ''; img.style.display = 'block'; };
  img.onerror = function () { st.textContent = ' Failed: 401 = wrong token, 409 = camera busy streaming.'; };
  img.src = withTok('/image?q=' + q + '&_=' + Date.now());
}
document.getElementById('token').addEventListener('input', refresh);
refresh();
</script>
</body>
</html>
"""


def build_home(base, image_tiers):
    host = base.split('://', 1)[-1]
    img_rows = ''.join(
        f'<tr><td><code>{k}</code></td><td>{w}&times;{h}</td><td>{q}</td>'
        f'<td><a data-path="/image?q={k}" href="/image?q={k}">open</a></td></tr>'
        for k, (w, h, q) in image_tiers.items())
    vid_rows = ''.join(
        f'<tr><td><code>{k}</code></td><td>{w}&times;{h}</td><td>{fps}</td><td>{br / 1e6:g} Mbps</td></tr>'
        for k, (w, h, fps, br) in VIDEO_TIERS.items())
    rec_rows = ''.join(
        f'<tr><td><code>{k}</code></td><td>{w}&times;{h}</td><td>{fps}</td><td>{br / 1e6:g} Mbps</td>'
        f'<td><a data-path="/record?q={k}&amp;t=10" href="/record?q={k}&amp;t=10">download 10 s</a></td></tr>'
        for k, (w, h, fps, br) in RECORD_TIERS.items())
    img_options = ''.join(
        f'<option value="{k}"{" selected" if k == "i3" else ""}>{k} ({w}&times;{h})</option>'
        for k, (w, h, q) in image_tiers.items())
    auth_note = ('This server requires a token.' if TOKEN
                 else 'This server has no token set, so you can leave this empty.')
    return (HOME_HTML
            .replace('@@BASE@@', base).replace('@@HOST@@', host)
            .replace('@@AUTH_NOTE@@', auth_note).replace('@@MAXT@@', str(MAX_RECORD_S))
            .replace('@@IMG_ROWS@@', img_rows).replace('@@VID_ROWS@@', vid_rows)
            .replace('@@REC_ROWS@@', rec_rows).replace('@@IMG_OPTIONS@@', img_options))


async def index(request):
    cm = request.app['cm']
    base = f'{request.scheme}://{request.host}'
    return web.Response(text=build_home(base, cm.image_tiers), content_type='text/html',
                        headers={'Cache-Control': 'no-store'})


async def favicon(request):
    return web.Response(status=204)


async def on_startup(app):
    app['cm'] = CameraManager(asyncio.get_running_loop())


async def on_cleanup(app):
    await app['cm'].shutdown()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    app = web.Application(middlewares=[auth])
    app.add_routes([web.get('/', index), web.get('/favicon.ico', favicon),
                    web.get('/options', options), web.get('/image', image),
                    web.get('/video', video), web.get('/record', record)])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host='0.0.0.0', port=PORT, access_log=None)


if __name__ == '__main__':
    main()
