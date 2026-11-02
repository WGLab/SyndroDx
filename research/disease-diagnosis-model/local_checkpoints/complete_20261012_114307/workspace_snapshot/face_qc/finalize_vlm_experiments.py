"""Bounded local postprocessing for the three running GPU experiments."""
import contextlib
import datetime
import io
import json
import time
from compare_vlm_experiments import main, QC

start = time.monotonic()
destination = QC / 'vlm_experiment_comparison'
destination.mkdir(exist_ok=True)
while True:
    with contextlib.redirect_stdout(io.StringIO()):
        main()
    summary = json.loads((destination / 'summary.json').read_text())
    finished = summary['all_complete']
    timed_out = time.monotonic() - start > 5400
    status = {'state': 'completed' if finished else ('postprocessing_timeout' if timed_out else 'running'),
              'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'counts': {k: summary[k]['processed'] for k in ['baseline', 'revised', 'expanded']},
              'quality_validated': False, 'training_admitted': 0}
    (destination / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    if finished or timed_out:
        break
    time.sleep(30)
