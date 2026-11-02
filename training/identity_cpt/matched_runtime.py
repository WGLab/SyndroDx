import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import *
ROOT=PROJECT/'transface_objective_comparison_20261028'
ID=CODE/'identity_inputs'
sys.path.insert(0,str(ID))
from control import BASE, WEIGHTS
GMDB=PROJECT/'gmdb_supervised_transface_20261026'
