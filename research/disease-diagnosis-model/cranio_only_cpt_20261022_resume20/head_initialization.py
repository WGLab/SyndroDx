"""Load only learned projection heads; never alter the original backbones."""
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def load_heads_only(model, source):
    state = dcpsd.get_model_state_dict(model)
    prefixes = tuple(f'{owner}.{head}.' for owner in ('student', 'teacher', 'model_ema')
                     for head in ('dino_head', 'ibot_head'))
    selected = {name: value for name, value in state.items() if name.startswith(prefixes)}
    if not selected or any('backbone' in name for name in selected):
        raise RuntimeError('head_only_loading_selection_invalid')
    reader = dcp.FileSystemReader(source)
    metadata = reader.read_metadata().state_dict_metadata
    # The original checkpoint may contain either alias of the EMA ModuleDict.
    source_names = {}
    for name in selected:
        choices = [name]
        if name.startswith('model_ema.'):
            choices.append('teacher.'+name[len('model_ema.'):])
        if name.startswith('teacher.'):
            choices.append('model_ema.'+name[len('teacher.'):])
        source_names[name] = next((candidate for candidate in choices if 'model.'+candidate in metadata), None)
        if source_names[name] is None:
            raise RuntimeError('required_head_tensor_missing')
    payload = {'model.'+source_names[name]: tensor for name, tensor in selected.items()}
    dcp.load(payload, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    for name in selected:
        state[name] = payload['model.'+source_names[name]]
    # Reapply the complete current state: backbone tensors were never loaded.
    dcpsd.set_model_state_dict(model, state)
    return len(selected)
