"""Ten active training days are a floor; queue/validation time do not count."""
import math
import time

MINIMUM_TRAINING_SECONDS = 10 * 24 * 3600
CONTINUE_STATES = {'paused_for_pool_expansion', 'segment_complete_unconverged',
                   'segment_complete_duration_pending', 'paused_walltime_checkpoint_preserved'}
COMPLETE_STATE = 'ten_training_days_and_loss_plateau_complete'


class ActiveTrainingClock:
    def __init__(self, credited=0., timer=time.monotonic):
        if not math.isfinite(credited) or credited < 0:
            raise ValueError('invalid_credited_training_time')
        self.credited = float(credited)
        self.timer = timer
        self.started = None

    def start(self):
        if self.started is None:
            self.started = self.timer()

    def pause(self):
        if self.started is not None:
            self.credited += max(0., self.timer() - self.started)
            self.started = None
        return self.credited


def completion(credited, plateau):
    if not isinstance(credited, (int, float)) or not math.isfinite(credited) or credited < 0:
        raise ValueError('invalid_credited_training_time')
    floor = credited >= MINIMUM_TRAINING_SECONDS
    return dict(complete=bool(floor and plateau), minimum_duration_met=floor,
                credited_training_seconds=credited,
                minimum_training_seconds=MINIMUM_TRAINING_SECONDS,
                remaining_minimum_training_seconds=max(0., MINIMUM_TRAINING_SECONDS-credited),
                operational_loss_plateau=bool(plateau))


def validation_policy(decision):
    """User requested continued SSL; finite probe regression is advisory."""
    result = dict(decision)
    if result.get('state') == 'paused_validation_regression':
        result.update(continue_training=True, state='warning_validation_regression_continuing',
                      original_guard_state='paused_validation_regression',
                      user_requested_minimum_ten_training_days=True)
    return result
