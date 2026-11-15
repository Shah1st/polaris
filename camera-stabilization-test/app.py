from __future__ import annotations

import argparse
import logging
import signal
from urllib.request import Request, urlopen

from stabilization.pipeline import StabilizationPipeline
from stabilization.source import FrameSource
from stabilization.stabilizer import VidStabAdapter
from web.server import WebServer

logger = logging.getLogger(__name__)


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description='Show raw and stabilized camera video.')
    parser.add_argument(
        '--source',
        default='auto',
        help='webcam index, "auto", or video file path',
    )
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=24,
        help='stabilization strength and delay; smaller is faster',
    )
    parser.add_argument(
        '--zoom',
        type=float,
        default=1.35,
        help='zoom before stabilization to hide transformed edges',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8085)
    parser.add_argument(
        '--stop',
        action='store_true',
        help='request a running app on this port to stop',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    if args.stop:
        url = f'http://{args.host}:{args.port}/api/stop'
        try:
            with urlopen(Request(url, method='POST'), timeout=3) as response:
                print(response.read().decode().strip())
        except OSError as exc:
            raise SystemExit(f'cannot stop app at {url}: {exc}') from exc
        return

    source = FrameSource(
        parse_source(args.source),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    pipeline = StabilizationPipeline(
        source,
        VidStabAdapter(
            smoothing_window=args.smoothing_window,
            zoom=args.zoom,
        ),
    )
    server = WebServer(pipeline, host=args.host, port=args.port)

    def stop(_signum, _frame) -> None:
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    pipeline.start()
    logger.info('open http://%s:%d', args.host, args.port)
    logger.info('stop with Ctrl+C or: .venv/bin/python app.py --stop --port %d', args.port)
    try:
        server.serve_forever()
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
