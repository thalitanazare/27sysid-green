"""E6c - tempo real por amostra das redes neurais em C (mesmo protocolo do E3b).

Le fase2_outputs/e6c/<caso>_<modelo>.json (gerado por e6c_nn_train.py), gera C
com os pesos do modelo escolhido, roda o free run da validacao e mede o tempo
por amostra (21 repeticoes de >= 50 ms, mediana). Mesmos perfis de compilacao
do E3b. Confere a saida contra o free run em float64 do Python.

Uso: python e6c_bench.py     Saida: fase2_outputs/e6c/e6c_timing.csv
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN = ROOT / "fase2_outputs" / "e6c"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
CASES = ["dc", "ced", "silverbox"]
MODELS = ["mlp", "lstm", "subnet"]
PROFILES = {"direto": ["-O2", "-ffp-contract=off"],
            "otimizado": ["-O2", "-march=native", "-ffp-contract=fast"]}


def carr(name, v):
    v = np.asarray(v, float).ravel()
    return f"static const double {name}[{len(v)}] = {{" + ",".join(repr(float(x)) for x in v) + "};"


def mlpb_c(prefix, P):
    """Declaracoes e corpo de uma rede densa tanh com bypass opcional: f(x, out)."""
    decl, body = [], []
    layers = P["layers"]
    nin = len(layers[0][0][0])
    for i, (W, b) in enumerate(layers):
        decl += [carr(f"{prefix}W{i}", W), carr(f"{prefix}b{i}", b)]
    if P["bypass"] is not None:
        decl += [carr(f"{prefix}Wl", P["bypass"][0]), carr(f"{prefix}bl", P["bypass"][1])]
    body.append(f"static void __attribute__((noinline)) {prefix}(const double *x0, double *out) {{")
    body.append("    double a[256], c[256]; const double *x = x0; double *y = a;")
    for i, (W, b) in enumerate(layers):
        no, ni = len(W), len(W[0])
        last = i == len(layers) - 1
        dst = "out" if last else "y"
        body.append(f"    for (int o = 0; o < {no}; o++) {{ double s = {prefix}b{i}[o]; "
                    f"for (int j = 0; j < {ni}; j++) s += {prefix}W{i}[o*{ni}+j]*x[j]; "
                    f"{dst}[o] = {'s' if last else 'tanh(s)'}; }}")
        if not last:
            body.append("    x = y; y = (y == a) ? c : a;")
    if P["bypass"] is not None:
        no = len(P["bypass"][0])
        body.append(f"    for (int o = 0; o < {no}; o++) {{ double s = {prefix}bl[o]; "
                    f"for (int j = 0; j < {nin}; j++) s += {prefix}Wl[o*{nin}+j]*x0[j]; out[o] += s; }}")
    body.append("}")
    return "\n".join(decl), "\n".join(body), nin


def gen(case, model, d):
    sc = d["scaling"]
    mu, su, my, sy = sc["mu"], sc["su"], sc["my"], sc["sy"]
    w = d["weights"]
    cfg = d["cfg"]
    lag = cfg["lag"]
    if model == "mlp":
        decl, body, nin = mlpb_c("net", w["P"])
        step = f"""
static void run(void) {{
    double x[{2 * lag}], o[1];
    for (long k = S0; k < N; k++) {{
        for (int i = 1; i <= {lag}; i++) {{ x[i-1] = (Y[k-i] - {my!r})*{1 / sy!r}; x[{lag}+i-1] = (U[k-i] - {mu!r})*{1 / su!r}; }}
        net(x, o); Y[k] = o[0]*{sy!r} + {my!r};
    }}
}}"""
        start = "S0"
    elif model == "lstm":
        st = w["state"]
        h = len(st["lstm.weight_hh_l0"][0])
        bias = np.asarray(st["lstm.bias_ih_l0"]) + np.asarray(st["lstm.bias_hh_l0"])
        decl = "\n".join([carr("Wi", st["lstm.weight_ih_l0"]), carr("Wh", st["lstm.weight_hh_l0"]), carr("bb", bias),
                          carr("Wo", st["out.weight"]), carr("bo", st["out.bias"])])
        warm = cfg["warm"]
        body = f"""
static double sig(double z) {{ return 1.0/(1.0+exp(-z)); }}
static void __attribute__((noinline)) cell(double uu, double *hs, double *cs) {{
    double g[{4 * h}];
    for (int r = 0; r < {4 * h}; r++) {{ double s = bb[r] + Wi[r]*uu; for (int j = 0; j < {h}; j++) s += Wh[r*{h}+j]*hs[j]; g[r] = s; }}
    for (int j = 0; j < {h}; j++) {{
        double ig = sig(g[j]), fg = sig(g[{h}+j]), gg = tanh(g[{2 * h}+j]), og = sig(g[{3 * h}+j]);
        cs[j] = fg*cs[j] + ig*gg; hs[j] = og*tanh(cs[j]);
    }}
}}"""
        s0 = "0" if warm is None else f"(S0 > {warm} ? S0 - {warm} : 0)"
        step = f"""
static void run(void) {{
    double hs[{h}] = {{0}}, cs[{h}] = {{0}};
    for (long k = {s0}; k < N; k++) {{
        cell((U[k] - {mu!r})*{1 / su!r}, hs, cs);
        double s = bo[0]; for (int j = 0; j < {h}; j++) s += Wo[j]*hs[j];
        Y[k] = s*{sy!r} + {my!r};
    }}
}}"""
        start = s0
    else:
        P = w["P"]
        parts = [mlpb_c(n, P[n]) for n in ("enc", "f", "h")]
        decl = "\n".join(p[0] for p in parts)
        body = "\n".join(p[1] for p in parts)
        nx = len(P["f"]["layers"][-1][0])
        step = f"""
static void run(void) {{
    double p[{2 * lag}], x[{nx + 1}], xn[{nx}], o[1];
    for (int i = 1; i <= {lag}; i++) {{ p[i-1] = (Y[S0-i] - {my!r})*{1 / sy!r}; p[{lag}+i-1] = (U[S0-i] - {mu!r})*{1 / su!r}; }}
    enc(p, x);
    for (long k = S0; k < N; k++) {{
        h(x, o); Y[k] = o[0]*{sy!r} + {my!r};
        x[{nx}] = (U[k] - {mu!r})*{1 / su!r};
        f(x, xn); for (int i = 0; i < {nx}; i++) x[i] = xn[i];
    }}
}}"""
        start = "S0"
    return decl, body, step, start


def write_c(case, model, d, u, y, s0):
    decl, body, step, start = gen(case, model, d)
    n = len(u)
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#define N {n}
#define S0 {s0}
{carr('U', u)}
{carr('YM', y)}
static double Y[N];
{decl}
{body}
{step}
static double now_s(void) {{ LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart/(double)f.QuadPart; }}
static int cmp(const void *a, const void *b) {{ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }}
int main(void) {{
    SetThreadAffinityMask(GetCurrentThread(), 1);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    long first = {start};
    memcpy(Y, YM, sizeof Y); run();
    printf("OUT"); for (long k = S0; k < N; k++) printf(" %.17g", Y[k]); printf("\\n");
    long reps = 1; double t;
    do {{ reps *= 2; memcpy(Y, YM, sizeof Y); t = now_s(); for (long r = 0; r < reps; r++) run(); t = now_s() - t; }} while (t < 0.05);
    double v[21];
    for (int r = 0; r < 21; r++) {{ memcpy(Y, YM, sizeof Y); double t0 = now_s(); for (long q = 0; q < reps; q++) run(); v[r] = (now_s() - t0) / ((double)reps * (N - first)); }}
    qsort(v, 21, sizeof(double), cmp);
    printf("TIME %.6f %.6f %.6f\\n", 1e9*v[10], 1e9*v[5], 1e9*v[15]);
    return 0;
}}
"""
    p = BUILD / f"nn_{case}_{model}.c"
    p.write_text(src, encoding="utf-8")
    return p


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from e6b_mlp_narx import load_case        # mesma preparacao dos dados
    rows = []
    for case in CASES:
        _, _, uv, yv, s0 = load_case(case)
        for model in MODELS:
            f = IN / f"{case}_{model}.json"
            if not f.exists():
                print("faltando", f.name)
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            src = write_c(case, model, d, uv, yv, s0)
            sel = d["selected_run"]
            yref = np.array([np.nan if v is None else v for v in (d["yhat_val"] or [])], float)
            for prof, flags in PROFILES.items():
                exe = BUILD / f"nn_{case}_{model}_{prof}.exe"
                subprocess.run(["gcc", *flags, "-o", str(exe), str(src), "-lm"], check=True)
                out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
                yc, tm = None, None
                for line in out.splitlines():
                    tok = line.split()
                    if tok[0] == "OUT":
                        yc = np.array(tok[1:], float)
                    elif tok[0] == "TIME":
                        tm = tuple(map(float, tok[1:]))
                diff = float(np.max(np.abs(yc - yref))) if yref.size == yc.size else float("nan")
                tgt = yv[s0:]
                nr = float(np.linalg.norm(tgt - yc) / np.linalg.norm(tgt - tgt.mean()))
                rows.append(dict(case=case, model=model, profile=prof, hp=json.dumps(sel["hp"]), A=sel["A"], M=sel["M"],
                                 activations=sel["act"], C=sel["C_karatsuba"], val_nrmse=nr,
                                 ns_median=tm[0], ns_q1=tm[1], ns_q3=tm[2], max_abs_diff_vs_python=diff,
                                 search_train_s=d["search_train_s"], selected_train_s=d["selected_train_s"]))
                print(f"{case:9s} {model:6s} {prof:9s} A/M={sel['A']}/{sel['M']} NRMSE={nr:.3f} "
                      f"{tm[0]:9.2f} ns  diff={diff:.1e}  treino(busca)={d['search_train_s']:.0f} s", flush=True)
    out = IN / "e6c_timing.csv"
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("Saida:", out)


if __name__ == "__main__":
    main()
