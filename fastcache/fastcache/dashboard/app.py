import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

logger = logging.getLogger("fastcache.dashboard")

# Global reference populated by serve_dashboard()
_global_cache = None

import math

def clean_floats(obj):
    if isinstance(obj, dict):
        return {k: clean_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
    return obj

def get_cache_data():
    """Retrieve cache data from global instance or temporary file."""
    if _global_cache:
        stats = _global_cache.stats.as_dict()
        entries = []
        # Attempt to grab entries if the store supports it (e.g., InMemoryStore)
        if hasattr(_global_cache.store, "_entries") and hasattr(_global_cache.store, "_lock"):
            with _global_cache.store._lock:
                for ns, ns_entries in _global_cache.store._entries.items():
                    for e in ns_entries:
                        if not e.is_expired:
                            entries.append({
                                "id": e.id,
                                "namespace": e.namespace,
                                "query": e.query,
                                "hit_count": e.hit_count,
                                "age_seconds": e.age_seconds,
                                "ttl": e.ttl,
                                "ttl_remaining": e.ttl_remaining if e.ttl > 0 else None,
                            })
        return clean_floats({"stats": stats, "entries": entries})
        
    path = os.environ.get("FASTCACHE_DEMO_CACHE")
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading demo cache file: {e}")
            
    return {"stats": {}, "entries": []}


class DashboardHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Suppress default noisy access logs
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            try:
                with open(html_path, "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.wfile.write(b"<h1>Dashboard Error</h1><p>index.html not found.</p>")
                logger.error(f"Failed to serve index.html: {e}")
                
        elif parsed.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            data = get_cache_data()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=5555):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"Dashboard running on http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    import sys
    port = 5555
    if "--port" in sys.argv:
        try:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        except (ValueError, IndexError):
            pass
            
    logging.basicConfig(level=logging.INFO)
    run_server(port)
