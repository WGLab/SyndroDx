"""Run the frozen VLM worker with a separately versioned prompt/input manifest."""
import argparse
from pathlib import Path
import vlm_screen as worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=worker.INPUT)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    worker.INPUT = args.input.resolve()
    worker.PROMPT = args.prompt.resolve()
    worker.run(args)
