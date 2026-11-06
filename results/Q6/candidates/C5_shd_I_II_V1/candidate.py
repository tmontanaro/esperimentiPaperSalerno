"""Candidato CNN generato da tools/cnn_cv.py make-candidate. Contratto: predict(df, lib) -> DataFrame[score, in_scope]."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "tools" / "cnn_cv.py").exists():
    ROOT = ROOT.parent
WEIGHTS = ROOT / "results/Q6/final_I_II_V1"
TARGET = "shd_moderate_or_greater_flag"


def predict(df, lib):
    sys.path.insert(0, str(ROOT / "tools"))
    import cnn_cv
    return cnn_cv.candidate_predict(df, lib, WEIGHTS, TARGET)
