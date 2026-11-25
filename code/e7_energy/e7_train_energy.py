"""E7c - energia medida do treino das redes e do NSGA-II (RAPL).

Roda, um de cada vez, cada busca de hiperparametros das redes (e6c_nn_train.py,
mesma grade e sementes do artigo, saida redirecionada para nao sobrescrever)
e uma semente do NSGA-II (e8_nsga2.py, modo 'energy'). Mede a energia do
processo inteiro (pacote + DRAM) e desconta o repouso medido logo antes.

Uso: python e7_train_energy.py [nn|nsga|all]
Saida: fase2_outputs/e7/e7_train_energy.csv (acrescenta linhas)
"""
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rapl import Rapl, bracketed

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
OUT = MAIN / "fase2_outputs" / "e7"
OUT.mkdir(parents=True, exist_ok=True)
VENV = sys.executable  # Python environment with PyTorch
N_IDLE = 10
LOCAL = Path(tempfile.gettempdir()) / "e7_e6c_rerun"  # fora do Drive (sem sincronizacao durante a medicao)


def run(r, label, cmd, env=None):
    """Protocolo v2: linha de base = mediana de N_IDLE janelas de 1 s antes e
    N_IDLE depois da carga (rapl.bracketed), robusta a picos curtos."""
    holder = {}

    def job():
        holder["p"] = subprocess.run(cmd, cwd=str(MAIN), env=env, capture_output=True, text=True)

    dyn, t, base, spread, prun, _ = bracketed(r, job, n_idle=N_IDLE)
    p = holder["p"]
    row = dict(workload=label, wall_s=t, dynamic_J=dyn, P_idle_W=base, P_run_W=prun,
               idle_spread_W=spread, ok=p.returncode == 0)
    print(f"{label:28s} {t:9.1f} s  {dyn:10.1f} J dinamica  (base {base:.1f} W, carga {prun:.1f} W, "
          f"spread {spread:.1f} W){'' if p.returncode == 0 else '  ERRO'}", flush=True)
    if p.returncode != 0:
        print(p.stderr[-800:])
    return row


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = Rapl()
    rows = []
    if which in ("nn", "all"):
        env = dict(os.environ, E6C_OUT=str(LOCAL))
        for case in ["dc", "ced", "silverbox"]:
            for model in ["mlp", "lstm", "subnet"]:
                rows.append(run(r, f"{case} {model}", [VENV, "e6c_nn_train.py", case, model], env))
    if which in ("nsga", "all"):
        for case in ["dc", "ced", "silverbox"]:
            rows.append(run(r, f"{case} nsga2 (1 semente)", [VENV, str(MAIN / "e8_moo" / "e8_nsga2.py"), case, "energy"]))
    out = OUT / "e7_train_energy_v2.csv"
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("Saida:", out)


if __name__ == "__main__":
    main()
