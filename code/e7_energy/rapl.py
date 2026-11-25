"""Leitura dos contadores RAPL do Windows (Energy Meter), sem administrador.

Instancias: RAPL_Package0_PKG (pacote), RAPL_Package0_PP0 (nucleos),
RAPL_Package0_PP1 (grafico integrado), RAPL_Package0_DRAM (memoria).
O contador 'Energy' e cumulativo. A unidade e determinada em calibrate().
"""
import time

import win32pdh

INSTANCES = ["RAPL_Package0_PKG", "RAPL_Package0_PP0", "RAPL_Package0_DRAM"]


class Rapl:
    def __init__(self):
        self.q = win32pdh.OpenQuery()
        self.h = {}
        for inst in INSTANCES:
            self.h[inst] = win32pdh.AddEnglishCounter(self.q, rf"\Energy Meter({inst})\Energy")

    def read(self):
        """Valores brutos cumulativos (unidade do contador) e o instante (s)."""
        win32pdh.CollectQueryData(self.q)
        t = time.perf_counter()
        out = {}
        for inst, h in self.h.items():
            _, val = win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_LARGE)
            out[inst] = val
        return t, out


PWH = 3.6e-9                 # J por picowatt-hora
KEYS = ("RAPL_Package0_PKG", "RAPL_Package0_DRAM")


def joules(e0, e1):
    """Energia (pacote + DRAM) entre duas leituras, em J."""
    return sum((e1[k] - e0[k]) * PWH for k in KEYS)


def idle_windows(r, n, w=1.0):
    """n janelas consecutivas de repouso de w segundos. Retorna as potencias (W)."""
    p = []
    t0, e0 = r.read()
    for _ in range(n):
        time.sleep(w)
        t1, e1 = r.read()
        p.append(joules(e0, e1) / (t1 - t0))
        t0, e0 = t1, e1
    return p


def bracketed(r, fn, n_idle=3, w=1.0):
    """Repouso (n_idle janelas) - carga fn() - repouso (n_idle janelas).

    Linha de base = mediana das 2*n_idle janelas, robusta a picos curtos de
    outros processos. spread = max - min das janelas (usado para rejeitar
    ciclos em que a linha de base mudou). Retorna (dyn_J, t_run, base_W,
    spread_W, P_run_W, resultado de fn)."""
    import statistics
    before = idle_windows(r, n_idle, w)
    t2, e2 = r.read()
    out = fn()
    t3, e3 = r.read()
    after = idle_windows(r, n_idle, w)
    win = before + after
    base = statistics.median(win)
    E = joules(e2, e3)
    t = t3 - t2
    return E - base * t, t, base, max(win) - min(win), E / t, out


if __name__ == "__main__":
    r = Rapl()
    t0, e0 = r.read()
    changes = []
    last = e0["RAPL_Package0_PKG"]
    tl = t0
    while time.perf_counter() - t0 < 3.0:
        t, e = r.read()
        if e["RAPL_Package0_PKG"] != last:
            changes.append(t - tl)
            last, tl = e["RAPL_Package0_PKG"], t
        time.sleep(0.002)
    t1, e1 = r.read()
    dt = t1 - t0
    print(f"intervalo {dt:.3f} s, atualizacoes do contador: {len(changes)}, periodo medio {1000*sum(changes)/max(len(changes),1):.1f} ms")
    for k in INSTANCES:
        d = e1[k] - e0[k]
        print(f"{k:20s} delta bruto {d:.4e}  -> se pWh: {d*3.6e-9/dt:6.2f} W   se pJ: {d*1e-12/dt:8.4f} W   se uJ: {d*1e-6/dt:10.2f} W")
