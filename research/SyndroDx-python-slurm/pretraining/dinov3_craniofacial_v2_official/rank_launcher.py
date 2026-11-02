import os,sys
from common import ROOT,PYTHON
rank=int(os.environ['CPT_RANK']);world=int(os.environ['CPT_WORLD'])
os.environ.update(RANK=str(rank),WORLD_SIZE=str(world),LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',MASTER_ADDR=os.environ['CPT_MASTER'],MASTER_PORT=os.environ['CPT_PORT'],TORCHELASTIC_RUN_ID='official-cpt-v2-'+os.environ['CPT_PORT'])
os.execv(str(PYTHON),[str(PYTHON),str(ROOT/'train_entry.py'),*sys.argv[1:]])
