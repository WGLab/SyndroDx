import argparse
from pathlib import Path
import stream_qwen_review_v3 as worker
worker.OUT = worker.ROOT / 'face_qc/qwen_prefetch_smoke'
worker.prepare(argparse.Namespace(batch_size=16))
worker.run(argparse.Namespace(worker=6, stop_after=64))
