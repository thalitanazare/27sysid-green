"""E3b - tempo real por amostra dos modelos em free run, em C.

Le fase2_outputs/e3b/<caso>_models.json (gerado por e3b_e4_export.m), gera
codigo C com o esquema de avaliacao direta do artigo (coeficiente vezes os
d_i fatores, soma dos p termos com p-1 adicoes, sem reuso de monomios),
compila e mede o tempo por amostra no free run da validacao.

Dois perfis de compilacao (sem -ffast-math nos dois):
  direto   : -O2 -ffp-contract=off   (operacoes exatamente como contadas)
  otimizado: -O2 -march=native -ffp-contract=fast (o compilador pode fundir
             multiplicacao e adicao em FMA, uma otimizacao realista)
Cada medicao: 21 repeticoes, cada uma com >= 50 ms de free run, mediana e IQR.
Afinidade num nucleo e prioridade alta. Confere a saida C contra o free run
do MATLAB.

Uso: python e3b_bench.py      Saidas: fase2_outputs/e3b/e3b_timing.csv
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN = ROOT / "fase2_outputs" / "e3b"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
CASES = ["dc", "ced", "silverbox"]
PROFILES = {"direto": ["-O2", "-ffp-contract=off"],
            "otimizado": ["-O2", "-march=native", "-ffp-contract=fast"]}


def term_expr(codes_row, theta):
    f = [int(c) for c in np.atleast_1d(codes_row) if int(c) > 0]
    e = f"({theta!r})"
    for c in f:
        sig, lag = divmod(c, 1000)
        e += f"*{'y' if sig == 1 else 'u'}[k-{lag}]"
    return e


def model_func(i, m):
    codes = np.atleast_2d(np.asarray(m["codes"], dtype=float))
    theta = np.atleast_1d(np.asarray(m["theta"], dtype=float))
    if codes.shape[0] != theta.size:
        codes = codes.T
    lines = [f"static double __attribute__((noinline)) f{i}(const double *restrict y, const double *restrict u, long k) {{"]
    for t, (row, th) in enumerate(zip(codes, theta)):
        e = term_expr(row, float(th))
        lines.append(f"    {'double acc = ' if t == 0 else 'acc += '}{e};")
    lines.append("    return acc;\n}")
    lag = int(max([int(c) % 1000 for c in codes.ravel() if c > 0] + [0]))
    return "\n".join(lines), lag


def write_c(case, data):
    models = data["models"]
    u = np.asarray(data["u"], float)
    y = np.asarray(data["y"], float)
    s0 = int(data["start0"])
    n = len(u)
    funcs, lags = zip(*(model_func(i, m) for i, m in enumerate(models)))
    arr = lambda name, v: f"static const double {name}[{len(v)}] = {{" + ",".join(repr(float(x)) for x in v) + "};"
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define N {n}
#define S0 {s0}
{arr('U', u)}
{arr('YM', y)}
static double Y[N];
{chr(10).join(funcs)}
typedef double (*fn)(const double *restrict, const double *restrict, long);
static fn F[] = {{{','.join(f'f{i}' for i in range(len(models)))}}};
static double now_s(void) {{ LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }}
static void run(fn f) {{ for (long k = S0; k < N; k++) Y[k] = f(Y, U, k); }}
static int cmp(const void *a, const void *b) {{ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }}
int main(void) {{
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    for (int m = 0; m < (int)(sizeof F/sizeof F[0]); m++) {{
        memcpy(Y, YM, sizeof Y);
        run(F[m]);
        printf("OUT %d", m); for (long k = S0; k < N; k++) printf(" %.17g", Y[k]); printf("\\n");
        long reps = 1; double t;
        do {{ reps *= 2; memcpy(Y, YM, sizeof Y); t = now_s(); for (long r = 0; r < reps; r++) run(F[m]); t = now_s() - t; }} while (t < 0.05);
        double v[21];
        for (int r = 0; r < 21; r++) {{ memcpy(Y, YM, sizeof Y); double t0 = now_s(); for (long q = 0; q < reps; q++) run(F[m]); v[r] = (now_s() - t0) / ((double)reps * (N - S0)); }}
        qsort(v, 21, sizeof(double), cmp);
        printf("TIME %d %.6f %.6f %.6f\\n", m, 1e9*v[10], 1e9*v[5], 1e9*v[15]);
    }}
    return 0;
}}
"""
    p = BUILD / f"bench_{case}.c"
    p.write_text(src, encoding="utf-8")
    return p


def main():
    rows = []
    for case in CASES:
        data = json.loads((IN / f"{case}_models.json").read_text(encoding="utf-8"))
        models = data["models"]
        if isinstance(models, dict):
            models = [models]
            data["models"] = models
        src = write_c(case, data)
        for prof, flags in PROFILES.items():
            exe = BUILD / f"bench_{case}_{prof}.exe"
            subprocess.run(["gcc", *flags, "-o", str(exe), str(src)], check=True)
            out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
            outs, times = {}, {}
            for line in out.splitlines():
                tok = line.split()
                if tok[0] == "OUT":
                    outs[int(tok[1])] = np.array(tok[2:], float)
                elif tok[0] == "TIME":
                    times[int(tok[1])] = tuple(map(float, tok[2:]))
            for i, m in enumerate(models):
                yh = np.atleast_1d(np.array([np.nan if v is None else v for v in np.atleast_1d(m["yhat"])], float))
                diff = float(np.max(np.abs(outs[i] - yh))) if np.all(np.isfinite(yh)) else float("nan")
                med, q1, q3 = times[i]
                rows.append(dict(case=case, profile=prof, label=m["label"],
                                 p=len(np.atleast_1d(m["theta"])), A=int(m["nAdd"]), M=int(m["nMult"]),
                                 C=round(float(m["C"]), 2), ns_median=med, ns_q1=q1, ns_q3=q3,
                                 max_abs_diff_vs_matlab=diff))
                print(f"{case:9s} {prof:9s} {m['label']:16s} A/M={m['nAdd']:3d}/{m['nMult']:3d} "
                      f"C={m['C']:10.1f}  {med:7.3f} ns [{q1:.3f},{q3:.3f}]  diff={diff:.1e}", flush=True)
    out = IN / "e3b_timing.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Relacao custo x tempo
    for prof in PROFILES:
        r = [x for x in rows if x["profile"] == prof]
        t = np.array([x["ns_median"] for x in r])
        for name, c in [("C Karatsuba", np.array([x["C"] for x in r])),
                        ("a+b (rho medido ~1)", np.array([x["A"] + x["M"] for x in r]))]:
            A = np.c_[c, np.ones_like(c)]
            coef, *_ = np.linalg.lstsq(A, t, rcond=None)
            r2 = 1 - np.sum((t - A @ coef) ** 2) / np.sum((t - t.mean()) ** 2)
            rho_s = spearmanr(c, t).statistic
            print(f"[{prof}] tempo ~ {name}: afim R2={r2:.3f}, Spearman={rho_s:.3f}, "
                  f"inclinacao={coef[0]:.3e} ns/unid, intercepto={coef[1]:.3f} ns")
    print("Saida:", out)


if __name__ == "__main__":
    main()
