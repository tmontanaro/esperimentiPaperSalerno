"""Verifica il contratto dei candidati Q6: predict(df, lib) su 40 righe di validation.
Stampa solo forma, indice coerente e intervallo (aggregati). Non usa il test."""
import sys, importlib.util
sys.path.insert(0, "lib")
import numpy as np
import echonext_lib as lib

df = lib.load_data().val.iloc[:40]
for name in ["C1_shd_all", "C2_lvef45_all", "C3_rv_all", "C4_as_all", "C5_shd_I_II_V1"]:
    d = f"results/Q6/candidates/{name}"
    spec = importlib.util.spec_from_file_location(name, f"{d}/candidate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = m.predict(df, lib)
    s = np.asarray(out["score"], dtype=float)
    ok = (len(out) == 40 and list(out.index) == list(df.index) and np.isfinite(s).all()
          and s.min() >= 0 and s.max() <= 1 and set(out.columns) >= {"score", "in_scope"})
    print(name, "righe", len(out), "colonne", list(out.columns), "contratto_ok", bool(ok),
          "score medio %.3f" % s.mean())
