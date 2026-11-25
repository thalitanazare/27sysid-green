"""E7a - energia medida por amostra na avaliacao dos modelos (RAPL).

Para cada modelo (18 polinomiais do E3b e 9 redes do E6c) gera um executavel
C que roda o free run da validacao repetidamente por T segundos num nucleo.
O harness le os contadores RAPL do Windows (pacote + DRAM) antes e depois,
mede o consumo em repouso logo antes (sem carga, sem leitura durante o
intervalo) e calcula a energia dinamica por amostra:

    E_dyn = (E_run - P_idle * t_run) / amostras

Cinco repeticoes intercaladas (repouso, carga), mediana e IQR.
Mesmo perfil de compilacao do E3b ("direto": -O2 -ffp-contract=off).
Rodar sem outras cargas pesadas no PC (o RAPL mede o pacote inteiro).

Protocolo v2: ciclos repouso-carga-repouso com linha de base robusta (rapl.bracketed).
Uso: python e7_eval_energy.py      Saida: fase2_outputs/e7/e7_eval_energy_v2.csv
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path.insert(0, str(MAIN / "e3_timing"))
sys.path.insert(0, str(MAIN))
import e3b_bench as pb      # noqa: E402  gerador dos polinomios
import e6c_bench as nb      # noqa: E402  gerador das redes
from rapl import Rapl, bracketed  # noqa: E402

OUT = MAIN / "fase2_outputs" / "e7"
OUT.mkdir(parents=True, exist_ok=True)
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
FLAGS = ["-O2", "-ffp-contract=off"]
T_RUN = 2.0
CYCLES = 8
MAX_SPREAD = 6.0             # W

HARNESS = r"""
static double now_s(void) {{ LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }}
int main(int argc, char **argv) {{
    int m = argc > 1 ? atoi(argv[1]) : 0;
    double T = argc > 2 ? atof(argv[2]) : 4.0;
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    long long passes = 0; double t0 = now_s(), t;
    do {{ memcpy(Y, YM, sizeof Y); {call}; passes++; t = now_s() - t0; }} while (t < T);
    printf("%lld %lld %.6f\n", passes, (long long)passes * (N - {first}), t);
    return 0;
}}
"""


def poly_source(case, data):
    models = data["models"]
    u = np.asarray(data["u"], float)
    y = np.asarray(data["y"], float)
    s0 = int(data["start0"])
    funcs = [pb.model_func(i, m)[0] for i, m in enumerate(models)]
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define N {len(u)}
#define S0 {s0}
static const double U[{len(u)}] = {{{",".join(repr(float(v)) for v in u)}}};
static const double YM[{len(y)}] = {{{",".join(repr(float(v)) for v in y)}}};
static double Y[N];
{chr(10).join(funcs)}
typedef double (*fn)(const double *restrict, const double *restrict, long);
static fn F[] = {{{','.join(f'f{i}' for i in range(len(models)))}}};
static void runm(int m) {{ for (long k = S0; k < N; k++) Y[k] = F[m](Y, U, k); }}
""" + HARNESS.format(call="runm(m)", first="S0")
    p = BUILD / f"e7_poly_{case}.c"
    p.write_text(src, encoding="utf-8")
    return p, [m["label"] for m in models], [dict(A=m["nAdd"], M=m["nMult"], C=m["C"]) for m in models]


def nn_source(case, model, d, u, y, s0):
    decl, body, step, start = nb.gen(case, model, d)
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#define N {len(u)}
#define S0 {s0}
{nb.carr('U', u)}
{nb.carr('YM', y)}
static double Y[N];
{decl}
{body}
{step}
""" + HARNESS.format(call="run()", first=start)
    p = BUILD / f"e7_nn_{case}_{model}.c"
    p.write_text(src, encoding="utf-8")
    return p


def compile_c(src):
    exe = src.with_suffix(".exe")
    subprocess.run(["gcc", *FLAGS, "-o", str(exe), str(src), "-lm"], check=True)
    return exe


def measure(r, cmd):
    """Protocolo v2: CYCLES ciclos repouso-carga-repouso (rapl.bracketed).
    Ciclos com spread da linha de base > MAX_SPREAD W sao rejeitados."""
    acc, all_, per_sample_ns = [], [], []
    for _ in range(CYCLES):
        dyn, t, base, spread, prun, out = bracketed(
            r, lambda: subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split())
        samples, t_loop = int(out[1]), float(out[2])
        rec = (dyn / samples, base, prun, prun - base)
        all_.append(rec)
        if spread <= MAX_SPREAD:
            acc.append(rec)
        per_sample_ns.append(1e9 * t_loop / samples)
    use = acc if len(acc) >= 3 else all_
    v = np.array([a[0] for a in use])
    return dict(nJ_median=1e9 * float(np.median(v)), nJ_q1=1e9 * float(np.percentile(v, 25)),
                nJ_q3=1e9 * float(np.percentile(v, 75)), dP_W=float(np.median([a[3] for a in use])),
                P_idle_W=float(np.median([a[1] for a in use])), P_run_W=float(np.median([a[2] for a in use])),
                accepted=len(acc), ns_per_sample=float(np.median(per_sample_ns)))


def main():
    r = Rapl()
    rows = []
    for case in ["dc", "ced", "silverbox"]:
        data = json.loads((MAIN / "fase2_outputs" / "e3b" / f"{case}_models.json").read_text(encoding="utf-8"))
        src, labels, ops = poly_source(case, data)
        exe = compile_c(src)
        for i, lab in enumerate(labels):
            m = measure(r, [str(exe), str(i), str(T_RUN)])
            rows.append(dict(case=case, family="polynomial", model=lab, **ops[i], **m))
            print(f"{case:9s} {lab:14s} {m['nJ_median']:9.2f} nJ/amostra  {m['ns_per_sample']:8.2f} ns  "
                  f"dP {m['dP_W']:.2f} W  base {m['P_idle_W']:.1f} W  aceitos {m['accepted']}/{CYCLES}", flush=True)
        u = np.asarray(data["u"], float); y = np.asarray(data["y"], float); s0 = int(data["start0"])
        for model in ["mlp", "lstm", "subnet"]:
            d = json.loads((MAIN / "fase2_outputs" / "e6c" / f"{case}_{model}.json").read_text(encoding="utf-8"))
            exe = compile_c(nn_source(case, model, d, u, y, s0))
            m = measure(r, [str(exe), "0", str(T_RUN)])
            sel = d["selected_run"]
            rows.append(dict(case=case, family="network", model=model.upper(), A=sel["A"], M=sel["M"],
                             C=sel["C_karatsuba"], **m))
            print(f"{case:9s} {model:14s} {m['nJ_median']:9.2f} nJ/amostra  {m['ns_per_sample']:8.2f} ns  "
                  f"dP {m['dP_W']:.2f} W  base {m['P_idle_W']:.1f} W  aceitos {m['accepted']}/{CYCLES}", flush=True)
    out = OUT / "e7_eval_energy_v2.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Saida:", out)


if __name__ == "__main__":
    main()
