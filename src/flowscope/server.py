"""Tiny live page: the training process publishes HTML frames, browsers get them over Server-Sent Events.

Stdlib only. The page keeps the last frame when the script exits and reconnects on its own when you
run the next script, so you can leave the tab open while you iterate.
"""
import json, threading

from flowscope.tour import JS as TOUR_JS
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlowScope</title>
<style>
  body{margin:0;background:#0b0e16;color:#d5dae6;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
  #bar{display:flex;gap:10px;align-items:center;padding:6px 14px;color:#6b7280;font-size:11px}
  #dot{width:8px;height:8px;border-radius:50%;background:#6b7280}
  #root{padding:0 8px 16px}
</style></head><body>
<div id="bar"><span id="dot"></span><span id="status">connecting...</span></div>
<div id="root"></div>
<div id="tour" style="padding:0 14px 24px"></div>
<script>%TOUR_JS%</script>
<script>
  const root = document.getElementById('root'), dot = document.getElementById('dot'), status = document.getElementById('status');
  const set = (c, t) => { dot.style.background = c; status.textContent = t; };
  const es = new EventSource('/events');
  es.onopen = () => set('#22d3ee', 'live');
  es.onmessage = e => {
    const m = JSON.parse(e.data);
    root.innerHTML = m.html;
    if (m.tour) FlowTour.attach(document.getElementById('tour'), m.tour);
    set('#22d3ee', 'live · updated ' + new Date().toLocaleTimeString());
  };
  es.onerror = () => set('#6b7280', 'run ended or not started · showing last frame · reconnects automatically');
</script></body></html>"""
PAGE = PAGE.replace("%TOUR_JS%", TOUR_JS.replace("</", "<\\/"))


def _lan_ip():
    """The address other devices on the LAN use to reach this machine (no packets are sent)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


class LiveServer:
    _shared = {}

    @classmethod
    def shared(cls, port=8765, host="127.0.0.1"):
        """One server per process and port, so several scopes (or reruns) reuse it."""
        if port not in cls._shared:
            cls._shared[port] = cls(port, host)
        return cls._shared[port]

    def __init__(self, port=8765, host="127.0.0.1"):
        self.frame, self.tour, self.version, self.cond = "", None, 0, threading.Condition()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/events":
                    return self._events()
                body = (PAGE if self.path in ("/", "/index.html") else server.frame).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.version != seen, timeout=15)
                            frame, tour, v = server.frame, server.tour, server.version
                        msg = f"data: {json.dumps(dict(html=frame, tour=tour))}\n\n" if v != seen else ": keepalive\n\n"
                        seen = v
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        for p in range(port, port + 20):  # next free port if something else holds this one
            try:
                self.httpd = ThreadingHTTPServer((host, p), Handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"FlowScope: no free port in {port}-{port + 19}")
        self.httpd.daemon_threads = True
        port = self.httpd.server_address[1]
        self.url = f"http://{host}:{port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"FlowScope live at {self.url}", flush=True)
        if host in ("0.0.0.0", ""):
            print(f"FlowScope: reachable from your network (e.g. Vision Pro) at http://{_lan_ip()}:{port}", flush=True)

    def publish(self, html, tour=None):
        with self.cond:
            self.frame, self.tour, self.version = html, tour, self.version + 1
            self.cond.notify_all()
