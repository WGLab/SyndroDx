"""Advisory metric policy: finite validation declines never stop training."""
def advisory(severe_collapse,repeated_declines,current_multiplier=1.):
    adverse=bool(severe_collapse or repeated_declines)
    return dict(continue_training=True,adverse_signal=adverse,
        multiplier=max(.01,current_multiplier*.5) if adverse else current_multiplier,
        action='reduce_backbone_lr_and_continue' if adverse else 'continue')
