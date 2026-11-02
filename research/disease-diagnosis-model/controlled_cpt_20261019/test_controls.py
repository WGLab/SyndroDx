"""Synthetic-only failure-path checks: no patient files or model weights read."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from configuration import configuration, WORLD, STEPS, BACKBONE_LR_RATIO
from guard import decide, KEYS
from head_initialization import load_heads_only
import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf
from dinov3.train import train as official


class Gates(unittest.TestCase):
    def setUp(self):
        self.original = dict.fromkeys(KEYS, .4)

    def test_each_endpoint_regression_pauses(self):
        for key in KEYS:
            candidate = dict(self.original, **{key:.399})
            self.assertFalse(decide(candidate, self.original, self.original)['continue_training'])

    def test_previous_checkpoint_regression_pauses_even_above_original(self):
        previous = dict.fromkeys(KEYS, .6)
        current = dict.fromkeys(KEYS, .5)
        self.assertFalse(decide(current, self.original, previous)['continue_training'])

    def test_equal_and_improvement_pass(self):
        self.assertTrue(decide(self.original, self.original, self.original)['continue_training'])
        self.assertTrue(decide(dict.fromkeys(KEYS, .5), self.original, self.original)['continue_training'])

    def test_missing_nonfinite_and_invalid_reference_stop(self):
        for value in (None, float('nan'), float('inf'), -.1, 1.1, True):
            for which in range(3):
                values = [dict(self.original) for _ in range(3)]
                values[which][KEYS[0]] = value
                self.assertEqual(decide(*values)['state'], 'paused_invalid_validation')


class Initialization(unittest.TestCase):
    @staticmethod
    def model():
        model = torch.nn.Module()
        def blocks():
            return torch.nn.ModuleDict({name:torch.nn.Linear(2, 2)
                for name in ('backbone', 'dino_head', 'ibot_head')})
        model.student, model.teacher = blocks(), blocks()
        model.model_ema = model.teacher
        return model

    def test_selective_loading_preserves_both_backbones_and_teacher_alias(self):
        source, target = self.model(), self.model()
        with torch.no_grad():
            for p in source.parameters(): p.fill_(9)
            for p in target.parameters(): p.fill_(1)
        with tempfile.TemporaryDirectory() as name:
            dcp.save({'model':source.state_dict()}, checkpoint_id=Path(name)/'synthetic')
            load_heads_only(target, Path(name)/'synthetic')
        for owner in ('student', 'teacher', 'model_ema'):
            for name, p in getattr(target, owner).named_parameters():
                expected = 1 if name.startswith('backbone.') else 9
                self.assertTrue(torch.equal(p, torch.full_like(p, expected)))

    def test_missing_head_cannot_silently_initialize(self):
        source, target = self.model(), self.model()
        state = {k:v for k,v in source.state_dict().items() if '.ibot_head.' not in k}
        with tempfile.TemporaryDirectory() as name:
            dcp.save({'model':state}, checkpoint_id=Path(name)/'synthetic')
            with self.assertRaisesRegex(RuntimeError, 'required_head_tensor_missing'):
                load_heads_only(target, Path(name)/'synthetic')


class Schedule(unittest.TestCase):
    def test_paired_configuration_identical_except_output(self):
        left, right = [OmegaConf.to_container(configuration(arm), resolve=True)
                       for arm in ('warmed_heads', 'random_heads')]
        del left['train']['output_dir']; del right['train']['output_dir']
        self.assertEqual(left, right)

    def test_bounded_schedule_and_conservative_backbone_lr(self):
        cfg = configuration('random_heads')
        with patch.object(official.distributed, 'get_world_size', return_value=WORLD):
            schedules = official.build_schedulers(cfg)
        self.assertEqual(len(schedules[0]), STEPS)
        self.assertAlmostEqual(max(schedules[0]), 5e-5)
        self.assertAlmostEqual(max(schedules[0])*BACKBONE_LR_RATIO, 1e-6)
        self.assertTrue(all(abs(t-.07)<1e-12 for t in schedules[3]))
        self.assertEqual(cfg.evaluation.eval_period_iterations, 0)
        self.assertEqual(cfg.checkpointing.period, 25)

    def test_official_freeze_then_backbone_ramp(self):
        groups = [dict(is_last_layer=False, is_frozen_backbone=is_backbone,
                       lr_multiplier=BACKBONE_LR_RATIO if is_backbone else 1,
                       wd_multiplier=1, params=[torch.nn.Parameter(torch.ones(1))])
                  for is_backbone in (True, False)]
        optimizer = torch.optim.AdamW(groups)
        official.apply_optim_scheduler(optimizer, 5e-5, .04, 5e-5, backbone_frozen=True)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0)
        official.apply_optim_scheduler(optimizer, 5e-5, .04, 5e-5, backbone_lr_scale=.5)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-7)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 5e-5)


if __name__ == '__main__':
    unittest.main()
