from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/"rddm_official_2p5d_meddino_condition_v1"
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))
