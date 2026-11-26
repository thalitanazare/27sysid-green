"""E8 - NSGA-II sobre subconjuntos de regressores, mesmos objetivos do Green.

Objetivos (minimizar), nos dados de identificacao:
  f1 = log10(L),  L = ||y - X_S theta||^2 / ||y||^2  (= 1 - ERR acumulado)
  f2 = C(S) = 64 (p - 1) + 64^1.585 sum d_i
Restricao: 2 <= |S| <= p_max (DC 15, CED 40, Silverbox 20). Mesmos candidatos
do artigo (grau e atrasos da Tabela 1). Uma thread. Populacao aleatoria (sem
semear com Green ou FROLS). L por minimos quadrados em X_S (o tempo de montar X
pre-calculo entra no tempo total).

Saidas: e8_moo/nsga2_<caso>.json com o tempo, as frentes finais por semente,
o NRMSE de validacao (free run) das solucoes da frente e as metricas de
comparacao com a varredura Green (green_sweep_<caso>.json).

Uso: <python do venv> e8_nsga2.py <dc|ced|silverbox> [check]
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent


def load_case(case):
    """Mesma preparacao dos dados do artigo (igual a e6b/e6c). start e 0-based."""
    if case == "dc":
        y = np.loadtxt(MAIN / "y_cc.csv").ravel()
        u = np.loadtxt(MAIN / "u_cc.csv").ravel()
        n = min(len(y), len(u))
        idx = np.arange(0, n, 500)
        y, u = y[idx], u[idx]
        nt = len(y) // 2
        return u[:nt], y[:nt], u, y, nt
    f = {"ced": MAIN / "CED_outputs" / "CED_experiment_2_data.mat",
         "silverbox": MAIN / "Silverbox_arrow_validation_outputs" / "Silverbox_arrow_full_data.mat"}[case]
    d = loadmat(f)
    n0 = int(np.asarray(d["n_initial"]).ravel()[0])
    return (d["u_train"].ravel().astype(float), d["y_train"].ravel().astype(float),
            d["u_test"].ravel().astype(float), d["y_test"].ravel().astype(float), n0)

CFG = {"dc": dict(deg=2, lag=2, pmax=15, gens=100, seeds=[0, 1, 2, 3, 4]),
       "ced": dict(deg=3, lag=10, pmax=40, gens=300, seeds=[0, 1, 2, 3, 4]),
       "silverbox": dict(deg=3, lag=10, pmax=20, gens=150, seeds=[0, 1, 2])}
POP = 100
RHO_MULT = 64 ** 1.585


# ------------------------------------------------------------- candidatos
def candidates(deg, lag):
    var = [1000 + j for j in range(1, lag + 1)] + [2000 + j for j in range(1, lag + 1)]
    rows = [()]
    for d in range(1, deg + 1):
        rows += list(combinations_with_replacement(var, d))
    return rows


def lagged(sig, code, lag, n):
    j = code % 1000
    return sig[lag - j: lag - j + n]


def build(u, y, rows, lag):
    n = len(y) - lag
    X = np.empty((n, len(rows)))
    for c, r in enumerate(rows):
        col = np.ones(n)
        for code in r:
            col = col * lagged(y if code < 2000 else u, code, lag, n)
        X[:, c] = col
    return X, y[lag:]


def key(r):
    return tuple(sorted(r))


# ------------------------------------------------------------- free run
def free_run(rows_sel, theta, u, y, start):
    ys = y.astype(float).copy()
    for k in range(start, len(y)):
        s = 0.0
        for r, th in zip(rows_sel, theta):
            v = th
            for code in r:
                j = code % 1000
                v *= ys[k - j] if code < 2000 else u[k - j]
            s += v
        ys[k] = s
        if not np.isfinite(s) or abs(s) > 1e6:
            return None
    return ys[start:]


def nrmse(t, yh):
    if yh is None:
        return float("inf")
    return float(np.linalg.norm(t - yh) / np.linalg.norm(t - t.mean()))


# ------------------------------------------------------------- problema
class Subset(Problem):
    """L por minimos quadrados (SVD) em X_S, robusto a colunas colineares."""

    def __init__(self, X, Y, deg_of, pmax):
        super().__init__(n_var=X.shape[1], n_obj=2, xl=0, xu=1, vtype=bool)
        self.X, self.Y, self.d, self.pmax = X, Y, deg_of, pmax
        self.yy = float(Y @ Y)

    def L(self, idx):
        Xs = self.X[:, idx]
        th = np.linalg.lstsq(Xs, self.Y, rcond=None)[0]
        r = self.Y - Xs @ th
        return max(float(r @ r) / self.yy, 1e-16)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.empty((len(X), 2))
        for i, x in enumerate(X):
            idx = np.flatnonzero(x)
            F[i, 0] = np.log10(self.L(idx))
            F[i, 1] = 64 * (len(idx) - 1) + RHO_MULT * self.d[idx].sum()
        out["F"] = F


class SubsetSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), bool)
        for i in range(n_samples):
            s = np.random.randint(2, problem.pmax + 1)
            X[i, np.random.choice(problem.n_var, s, replace=False)] = True
        return X


class SubsetCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        _, n_mat, n_var = X.shape
        Y = np.zeros_like(X)
        for k in range(n_mat):
            a, b = np.flatnonzero(X[0, k]), np.flatnonzero(X[1, k])
            U = np.union1d(a, b)
            lo, hi = sorted((len(a), len(b)))
            for c in range(2):
                s = int(np.clip(np.random.randint(lo, hi + 1), 2, min(problem.pmax, len(U))))
                Y[c, k, np.random.choice(U, s, replace=False)] = True
        return Y


class SubsetMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for _ in range(1 + np.random.geometric(0.5) - 1):
                on, off = np.flatnonzero(X[i]), np.flatnonzero(~X[i])
                ops = (["swap"] if len(off) > 0 else []) + \
                      (["add"] if len(on) < problem.pmax and len(off) > 0 else []) + \
                      (["del"] if len(on) > 2 else [])
                op = np.random.choice(ops)
                if op in ("swap", "del"):
                    X[i, np.random.choice(on)] = False
                if op in ("swap", "add"):
                    X[i, np.random.choice(off)] = True
        return X


class Dup(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(a.X, b.X)


def nondominated(P):
    P = np.asarray(P)
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        keep[i] = not np.any(np.all(P <= P[i], 1) & np.any(P < P[i], 1))
    return keep


def main():
    case = sys.argv[1]
    check_only = len(sys.argv) > 2 and sys.argv[2] == "check"
    cfg = CFG[case]
    ut, yt, uv, yv, start = load_case(case)
    rows = candidates(cfg["deg"], cfg["lag"])
    col = {key(r): i for i, r in enumerate(rows)}
    deg_of = np.array([len(r) for r in rows], float)

    t0 = time.perf_counter()
    X, Y = build(ut, yt, rows, cfg["lag"])
    t_pre = time.perf_counter() - t0
    prob = Subset(X, Y, deg_of, cfg["pmax"])

    def solve_theta(idx):
        return np.linalg.lstsq(X[:, idx], Y, rcond=None)[0]

    # verificacao contra a varredura Green (L, C e NRMSE do artigo)
    gs = json.loads((HERE / f"green_sweep_{case}.json").read_text())
    green = []
    for g in gs["green"]:
        codes = np.atleast_2d(np.asarray(g["codes"], float)).astype(int)
        idx = [col[key(tuple(c for c in r if c > 0))] for r in codes]
        green.append(dict(wE=g["wE"], idx=idx, L=g["L"], C=g["C"], nrmse=g["nrmse"]))
    for g in green:
        if abs(g["wE"] - 0.7) < 1e-9:
            Lp = prob.L(np.array(g["idx"]))
            th = solve_theta(g["idx"])
            nr = nrmse(yv[start:], free_run([rows[i] for i in g["idx"]], th, uv, yv, start))
            print(f"verificacao w_E=0.70: L python {Lp:.6e} x artigo {g['L']:.6e} | NRMSE python {nr:.4f} x artigo {g['nrmse']:.4f}", flush=True)
    print(f"pre-calculo (matriz de candidatos): {t_pre:.1f} s, {X.shape[1]} candidatos", flush=True)
    if check_only:
        return
    if len(sys.argv) > 2 and sys.argv[2] == "energy":
        # modo de medicao de energia (E7c): uma semente, sem analise nem saida
        np.random.seed(cfg["seeds"][0])
        alg = NSGA2(pop_size=POP, sampling=SubsetSampling(), crossover=SubsetCrossover(),
                    mutation=SubsetMutation(), eliminate_duplicates=Dup())
        t1 = time.perf_counter()
        minimize(prob, alg, ("n_gen", cfg["gens"]), seed=cfg["seeds"][0], verbose=False)
        print(f"ENERGY_RUN {time.perf_counter() - t1 + t_pre:.3f}", flush=True)
        return

    runs = []
    for seed in cfg["seeds"]:
        np.random.seed(seed)
        alg = NSGA2(pop_size=POP, sampling=SubsetSampling(), crossover=SubsetCrossover(),
                    mutation=SubsetMutation(), eliminate_duplicates=Dup())
        t1 = time.perf_counter()
        res = minimize(prob, alg, ("n_gen", cfg["gens"]), seed=seed, verbose=False)
        t_run = time.perf_counter() - t1
        Xf = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        sols = [dict(idx=np.flatnonzero(x).tolist(), L=float(10 ** f[0]), C=float(f[1])) for x, f in zip(Xf, F)]
        runs.append(dict(seed=seed, time_s=t_run + t_pre, search_s=t_run, n_evals=int(res.algorithm.evaluator.n_eval),
                         front=sols))
        print(f"semente {seed}: {t_run:.1f} s de busca, {len(sols)} solucoes na frente", flush=True)

    # metricas de comparacao em (log10 L, C)
    Gp = np.array([[np.log10(max(g["L"], 1e-16)), g["C"]] for g in green])
    Gnd = Gp[nondominated(Gp)]
    allN = np.array([[np.log10(s["L"]), s["C"]] for r in runs for s in r["front"]])
    ref = np.array([max(Gp[:, 0].max(), allN[:, 0].max()) + 0.1, 1.05 * max(Gp[:, 1].max(), allN[:, 1].max())])
    lo = np.array([min(Gp[:, 0].min(), allN[:, 0].min()), 0.0])
    scale = ref - lo
    hv = HV(ref_point=np.ones(2))
    hv_g = float(hv((Gnd - lo) / scale))
    hv_n = [float(hv((np.array([[np.log10(s["L"]), s["C"]] for s in r["front"]]) - lo) / scale)) for r in runs]
    dom = []
    for g in green:
        p = np.array([np.log10(max(g["L"], 1e-16)), g["C"]])
        tol = 1e-9 * np.abs(p)                       # mesmo subconjunto recalculado difere so por arredondamento
        d = bool(np.any(np.all(allN <= p + tol, 1) & np.any(allN < p - tol, 1)))
        dom.append(dict(wE=g["wE"], dominated=d))

    # NRMSE de validacao das solucoes NSGA-II nao dominadas (uniao das sementes)
    pool = {}
    for r in runs:
        for s in r["front"]:
            pool[tuple(s["idx"])] = s
    U = list(pool.values())
    P = np.array([[np.log10(s["L"]), s["C"]] for s in U])
    U = [s for s, k in zip(U, nondominated(P)) if k]
    for s in U:
        th = solve_theta(s["idx"])
        s["nrmse"] = nrmse(yv[start:], free_run([rows[i] for i in s["idx"]], th, uv, yv, start))
        s["p"] = len(s["idx"])
        s["terms"] = [" ".join(str(c) for c in rows[i]) or "1" for i in s["idx"]]
    # comparacao com os tres representativos: melhor NSGA-II com C <= C_Green
    rep = []
    for g in green:
        if any(abs(g["wE"] - w) < 1e-9 for w in (0.3, 0.5, 0.7)):
            cand = [s for s in U if s["C"] <= g["C"] + 1e-6]
            bestL = min(cand, key=lambda s: s["L"]) if cand else None
            bestN = min(cand, key=lambda s: s["nrmse"]) if cand else None
            rep.append(dict(wE=g["wE"], green_L=g["L"], green_C=g["C"], green_nrmse=g["nrmse"],
                            nsga_bestL_L=None if bestL is None else bestL["L"],
                            nsga_bestL_nrmse=None if bestL is None else bestL["nrmse"],
                            nsga_bestNRMSE=None if bestN is None else bestN["nrmse"]))
    out = dict(case=case, pop=POP, gens=cfg["gens"], seeds=cfg["seeds"], pre_s=t_pre, runs=runs,
               hv_green=hv_g, hv_nsga=hv_n, green_dominated=dom, nsga_union_front=U, representative=rep)
    if case == "dc":
        ex = json.loads((HERE / "dc_exhaustive.json").read_text())
        exact = np.column_stack([np.log10(ex["frontL"]), ex["frontC"]])
        found = [bool(np.any(np.all(np.isclose(allN, e, rtol=1e-9, atol=1e-9), 1))) for e in exact]
        out["dc_exact_front_found"] = sum(found)
        out["dc_exact_front_size"] = len(found)
        hv_exact = float(hv((exact - lo) / scale))
        out["hv_exact"] = hv_exact
    (HERE / f"nsga2_{case}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"HV Green {hv_g:.4f} | HV NSGA-II {np.round(hv_n, 4)} | Green dominados: "
          f"{sum(d['dominated'] for d in dom)}/{len(dom)}", flush=True)
    if case == "dc":
        print(f"frente exata encontrada: {out['dc_exact_front_found']}/{out['dc_exact_front_size']}, HV exata {out['hv_exact']:.4f}")
    for r in rep:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
