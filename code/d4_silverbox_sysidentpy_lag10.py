"""D4 - SysIdentPy no Silverbox com o mesmo conjunto de candidatos do Green-Box.

Mesma configuracao do notebook Silverbox_arrow_validation_SysIdentPy.ipynb
(FROLS, grau 3, err_tol = 0.9999, n_terms = 80, LeastSquares, sem selecao
de ordem), mudando so o lag: 10 (igual ao Green-Box) e 14 (checagem de
reproducao da Tabela 4: 10 termos, NRMSE 0.077).

Dados: Silverbox_arrow_full_data.mat exportado pelo notebook (hash conferido
contra o manifesto). Saidas em D4_outputs/.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.io import loadmat, savemat
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares

HERE = Path(__file__).resolve().parent
DATA = HERE / "Silverbox_arrow_validation_outputs" / "Silverbox_arrow_full_data.mat"
MANIFEST = HERE / "Silverbox_arrow_validation_outputs" / "Silverbox_run_manifest.json"
OUT = HERE / "D4_outputs"
OUT.mkdir(exist_ok=True)

BITS, ALPHA = 64, 1.585
CASES = {"lag10": 10, "lag14": 14}


def complexity(codes):
    """A/M e C com a mesma contagem do artigo (coeficiente incluido)."""
    codes = np.asarray(codes)
    n_add = max(len(codes) - 1, 0)
    n_mult = int(np.sum(codes > 0))
    return n_add, n_mult, n_add * BITS + n_mult * BITS**ALPHA


def names(codes):
    out = []
    for row in np.asarray(codes):
        f = [f"{'y' if c // 1000 == 1 else 'u'}(k-{c % 1000})" for c in map(int, row) if c]
        out.append(" * ".join(f) if f else "1")
    return out


d = loadmat(DATA)
man = json.loads(MANIFEST.read_text(encoding="utf-8"))
for k in ["u_train", "y_train", "u_test", "y_test"]:
    h = hashlib.sha256(np.asarray(d[k], dtype="<f8").tobytes()).hexdigest()
    assert h == man["data_hashes"]["arrow_full"][k], f"hash diferente em {k}"
u_tr, y_tr = d["u_train"].reshape(-1, 1), d["y_train"].reshape(-1, 1)
u_te, y_te = d["u_test"].reshape(-1, 1), d["y_test"].reshape(-1, 1)
n0 = int(np.asarray(d["n_initial"]).ravel()[0])

rows = []
for key, lag in CASES.items():
    model = FROLS(xlag=lag, ylag=lag, basis_function=Polynomial(degree=3),
                  estimator=LeastSquares(), err_tol=0.9999, n_terms=80,
                  order_selection=False)
    t0 = time.perf_counter()
    model.fit(X=u_tr, y=y_tr)
    fit_s = time.perf_counter() - t0
    ml = int(model.max_lag)
    start = n0 - ml
    yhat = np.asarray(model.predict(X=u_te[start:], y=y_te[start:n0]))[ml:].reshape(-1, 1)
    target = y_te[n0:]
    finite = bool(np.isfinite(yhat).all())
    res = target - yhat
    rmse = float(np.sqrt(np.mean(res**2))) if finite else float("inf")
    nrmse = rmse / float(np.std(target)) if finite else float("inf")
    codes = np.asarray(model.final_model, dtype=np.int64)
    theta = np.asarray(model.theta).ravel()
    n_add, n_mult, C = complexity(codes)
    row = dict(case=key, lag=lag, max_lag=ml, n_terms=len(theta), A=n_add, M=n_mult,
               C=round(C, 2), RMSE_mV=1000 * rmse, NRMSE=nrmse, R2=1 - nrmse**2,
               fit_seconds=fit_s)
    rows.append(row)
    print(json.dumps(row))
    for r, v in zip(names(codes), theta):
        print(f"   {v:+.6e}  {r}")
    savemat(OUT / f"Silverbox_sysidentpy_degree3_{key}_model.mat",
            dict(model_code=codes, theta=theta.reshape(-1, 1), yhat_test=yhat,
                 n_initial=n0, **{k: v for k, v in row.items() if k != "case"}))

(OUT / "D4_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("Checagem lag14 (artigo): 10 termos, 9/11, C = 8596.25, NRMSE 0.077")
