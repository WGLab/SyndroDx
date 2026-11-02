import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import *
WORK=PROJECT/'transface_arcface_identity_cpt_20261026'
BASE=Path(os.environ['CRL_DEVELOPMENT_INPUTS'])
WEIGHTS=Path(os.environ['TRANSFACE_WEIGHTS'])
