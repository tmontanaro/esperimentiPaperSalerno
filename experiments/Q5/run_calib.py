"""Rilancia calib_ops.py con i nomi corretti delle etichette RV e stenosi aortica (nomi di lib.FLAG_COLS)."""
from pathlib import Path
d = Path(__file__).resolve().parent
src = (d / "calib_ops.py").read_text()
src = src.replace('"rv_systolic_dysfunction_flag", "aortic_stenosis_flag"',
                  '"rv_systolic_dysfunction_moderate_or_greater_flag", "aortic_stenosis_moderate_or_greater_flag"')
(d / "calib_ops.py").write_text(src)
f = d.parents[0] / "results/Q5/calib_ops.json"
f = Path(__file__).resolve().parents[2] / "results/Q5/calib_ops.json"
if f.exists():
    f.unlink()
exec(compile(src, str(d / "calib_ops.py"), "exec"), {"__file__": str(d / "calib_ops.py")})
