"""Download one big file with many parallel HTTP range requests (resumable).

    python scripts/fetch_parallel.py <url> <out file> [connections] [chunk MB]

The hackathon Wi-Fi caps each connection (~0.3 MB/s from Hugging Face) but not
the total, so splitting into chunks fetched in parallel is several times faster.
"""
import concurrent.futures as cf
import os
import sys
import time
import urllib.request

url, out = sys.argv[1], sys.argv[2]
conns = int(sys.argv[3]) if len(sys.argv) > 3 else 16
chunk = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else 32_000_000
UA = {"User-Agent": "rovermap/0.1"}

req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
with urllib.request.urlopen(req, timeout=60) as r:
    total = int(r.headers["Content-Range"].split("/")[-1])
parts_dir = out + ".parts"
os.makedirs(parts_dir, exist_ok=True)
ranges = [(i, s, min(s + chunk, total) - 1) for i, s in enumerate(range(0, total, chunk))]
print(f"{total / 1e9:.2f} GB in {len(ranges)} chunks, {conns} connections", flush=True)


def fetch(job):
    i, s, e = job
    p = os.path.join(parts_dir, f"{i:05d}")
    if os.path.exists(p) and os.path.getsize(p) == e - s + 1:
        return 0
    # Generous retries: Wi-Fi / DNS outages of several minutes must not kill
    # a multi-hour download (IncompleteRead is not an OSError, hence Exception).
    for attempt in range(40):
        try:
            rq = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(rq, timeout=120) as r:
                data = r.read()
            if len(data) != e - s + 1:
                raise OSError("short read")
            with open(p + ".tmp", "wb") as f:
                f.write(data)
            os.replace(p + ".tmp", p)
            return len(data)
        except Exception:
            time.sleep(min(60, 2 + 3 * attempt))
    raise RuntimeError(f"chunk {i} failed")


t0, done = time.time(), 0
with cf.ThreadPoolExecutor(conns) as ex:
    for n in ex.map(fetch, ranges):
        done += n
        if n:
            el = time.time() - t0
            print(f"\r{done / 1e9:.2f} GB fetched this run, {done / 1e6 / max(el, 1e-3):.1f} MB/s", end="", flush=True)
with open(out + ".tmp", "wb") as f:
    for i, _, _ in ranges:
        with open(os.path.join(parts_dir, f"{i:05d}"), "rb") as pf:
            f.write(pf.read())
os.replace(out + ".tmp", out)
for i, _, _ in ranges:
    os.remove(os.path.join(parts_dir, f"{i:05d}"))
os.rmdir(parts_dir)
print(f"\ndone: {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {(time.time() - t0) / 60:.1f} min", flush=True)
