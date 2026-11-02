import sys
sys.path.insert(0,'/opt/craniofacial/SyndroDx-python-slurm/pretraining/dinov3_craniofacial_v1')
from runtime import ROOT,load,notify
x=load(ROOT/'training/diagnostics/full_export_audit.json')
if x.get('state')=='passed':
 notify('Full encoder export audit completed','Encoder integrity audit completed','The full local teacher-checkpoint to encoder-export integrity audit completed. This checks file integrity only, not representation quality. All private weights and data remain on HPC.')
else:
 notify('Full encoder export audit requires attention','Encoder integrity audit requires attention','The full local encoder integrity audit did not complete successfully. Existing models and checkpoints are preserved; private diagnostics remain on HPC.')
