"""E6c - redes neurais treinadas no mesmo PC, mesmos dados e protocolo.

Tres familias, todas em PyTorch (CPU, 1 thread, igual ao E4 e a Tabela 1 do artigo):
  mlp    : MLP-NARX, regressores y(k-1..k-n), u(k-1..k-n) com o mesmo atraso
           maximo dos candidatos Green-Box, treino um passo a frente
  lstm   : LSTM com entrada u, treino por simulacao em subsequencias
  subnet : SUBNET (Beintema et al. 2021): encoder psi(passado) -> x_k,
           x_{k+1} = f(x_k, u_k), y_k = h(x_k), redes 2 x 64 tanh com
           bypass linear, treino por simulacao em subsequencias

Protocolo
  * so os dados de identificacao entram no treino. Os ultimos 20 % deles
    sao reservados (holdout) para parada antecipada e escolha da
    configuracao, sempre por free run. A validacao do artigo so e usada
    para reportar, nunca para escolher.
  * grade pequena de hiperparametros x 3 sementes. O custo de treino
    reportado inclui TODA a busca (como no E4 para o Green).
  * free run de validacao em float64 com os pesos exportados, mesma
    inicializacao dos scripts MATLAB, NRMSE = ||e|| / ||y - media(y)||.
  * operacoes por amostra contadas como no E6 (densa n_in x n_out:
    n_in*n_out mult e n_in*n_out adicoes com o bias), mais escalonamento
    das entradas e desescalonamento da saida. Ativacoes a parte.
  * carbono do treino: tempo x taxa da Tabela 1 (121.81/60 mg CO2e/s,
    1 nucleo do i5-8500, Irlanda, Green Algorithms).

Uso: <python do venv> e6c_nn_train.py <dc|ced|silverbox> <mlp|lstm|subnet> [quick]
Saida: fase2_outputs/e6c/<caso>_<modelo>.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("E6C_OUT", HERE / "fase2_outputs" / "e6c"))   # E7c redireciona a saida
OUT.mkdir(parents=True, exist_ok=True)
RATE_MG_S = 121.81 / 60.0            # Tabela 1 do artigo, 1 nucleo
SEEDS = [0, 1, 2]

CFG = {
    #            lag nx  T_sub T_lstm  it_mlp it_lstm it_sub  ev   warm(lstm)
    "dc":        dict(lag=2,  nx=2, T_sub=30, T_lstm=60,  it_mlp=6000,  it_lstm=4000, it_sub=4000, ev=200, warm=None),
    "ced":       dict(lag=10, nx=3, T_sub=30, T_lstm=60,  it_mlp=6000,  it_lstm=4000, it_sub=4000, ev=200, warm=10),
    "silverbox": dict(lag=10, nx=4, T_sub=80, T_lstm=200, it_mlp=20000, it_lstm=6000, it_sub=8000, ev=250, warm=50),
}
GRID = {"mlp": [dict(depth=1, width=64), dict(depth=2, width=64)],
        "lstm": [dict(hidden=16), dict(hidden=64)],
        "subnet": [dict(depth=2, width=64)]}
PATIENCE = 10


# ------------------------------------------------------------------ dados
def load_case(case):
    """ut, yt, uv, yv, start (0-based, primeira amostra prevista na validacao)."""
    if case == "dc":
        y = np.loadtxt(HERE / "y_cc.csv").ravel()
        u = np.loadtxt(HERE / "u_cc.csv").ravel()
        n = min(len(y), len(u))
        idx = np.arange(0, n, 500)
        y, u = y[idx], u[idx]
        nt = len(y) // 2
        return u[:nt], y[:nt], u, y, nt
    f = {"ced": HERE / "CED_outputs" / "CED_experiment_2_data.mat",
         "silverbox": HERE / "Silverbox_arrow_validation_outputs" / "Silverbox_arrow_full_data.mat"}[case]
    d = loadmat(f)
    n0 = int(np.asarray(d["n_initial"]).ravel()[0])
    return (d["u_train"].ravel().astype(float), d["y_train"].ravel().astype(float),
            d["u_test"].ravel().astype(float), d["y_test"].ravel().astype(float), n0)


def nrmse(y, yh):
    if yh is None or not np.all(np.isfinite(yh)):
        return float("inf")
    return float(np.linalg.norm(y - yh) / np.linalg.norm(y - y.mean()))


# ------------------------------------------------------------------ redes
class MLPb(nn.Module):
    def __init__(self, nin, nout, width, depth, bypass):
        super().__init__()
        layers, d = [], nin
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, nout))
        self.net = nn.Sequential(*layers)
        self.lin = nn.Linear(nin, nout) if bypass else None

    def forward(self, x):
        out = self.net(x)
        return out + self.lin(x) if self.lin is not None else out


def mlpb_numpy(m):
    Ls = [l for l in m.net if isinstance(l, nn.Linear)]
    W = [(l.weight.detach().double().numpy(), l.bias.detach().double().numpy()) for l in Ls]
    B = None if m.lin is None else (m.lin.weight.detach().double().numpy(), m.lin.bias.detach().double().numpy())
    return {"layers": W, "bypass": B}


def mlpb_eval(P, x):
    h = x
    for i, (W, b) in enumerate(P["layers"]):
        h = W @ h + b
        if i < len(P["layers"]) - 1:
            h = np.tanh(h)
    if P["bypass"] is not None:
        h = h + P["bypass"][0] @ x + P["bypass"][1]
    return h


def mlpb_ops(nin, nout, width, depth, bypass):
    sizes = [nin] + [width] * depth + [nout]
    m = sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    a = m
    if bypass:
        m += nin * nout
        a += nin * nout + nout
    return a, m, width * depth


def jsonable(P):
    return {"layers": [[W.tolist(), b.tolist()] for W, b in P["layers"]],
            "bypass": None if P["bypass"] is None else [P["bypass"][0].tolist(), P["bypass"][1].tolist()]}


# ------------------------------------------------------------------ MLP-NARX
def narx_X(u, y, lag):
    X = np.column_stack([y[lag - i:len(y) - i] for i in range(1, lag + 1)] +
                        [u[lag - i:len(u) - i] for i in range(1, lag + 1)])
    return X, y[lag:]


def mlp_sim(P, sc, u, y, start, lag):
    mu, su, my, sy = sc
    ys = y.astype(float).copy()
    for k in range(start, len(y)):
        x = np.r_[(ys[k - lag:k][::-1] - my) / sy, (u[k - lag:k][::-1] - mu) / su]
        ys[k] = mlpb_eval(P, x)[0] * sy + my
        if not np.isfinite(ys[k]) or abs(ys[k]) > 1e6 * (abs(my) + sy):
            return None
    return ys[start:]


def train_mlp(case, cfg, hp, seed, data):
    ut, yt, nfit, sc = data["ut"], data["yt"], data["nfit"], data["sc"]
    mu, su, my, sy = sc
    lag = cfg["lag"]
    X, Y = narx_X((ut[:nfit] - mu) / su, (yt[:nfit] - my) / sy, lag)
    X, Y =torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)[:, None]
    torch.manual_seed(seed)
    net = MLPb(2 * lag, 1, hp["width"], hp["depth"], bypass=False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bs = min(256, len(X))
    rng = np.random.default_rng(seed)
    best, bestP, bad = np.inf, None, 0
    for it in range(1, cfg["it_mlp"] + 1):
        idx = torch.as_tensor(rng.integers(0, len(X), bs))
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % cfg["ev"] == 0:
            P = mlpb_numpy(net)
            e = nrmse(yt[nfit:], mlp_sim(P, sc, ut, yt, nfit, lag))
            if e < best:
                best, bestP, bad = e, P, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if bestP is None:
        bestP = mlpb_numpy(net)
    sim = lambda u, y, s: mlp_sim(bestP, sc, u, y, s, lag)
    a, m, act = mlpb_ops(2 * lag, 1, hp["width"], hp["depth"], False)
    a += 2 * lag + 1          # escalonamento (2 lag subtracoes) + desescalonamento
    m += 2 * lag + 1
    return best, it, sim, dict(A=a, M=m, act=f"{act} tanh"), {"P": jsonable(bestP)}


# ------------------------------------------------------------------ LSTM
class LSTMnet(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.lstm = nn.LSTM(1, h, batch_first=True)
        self.out = nn.Linear(h, 1)

    def forward(self, u):
        return self.out(self.lstm(u)[0])


def lstm_sim(net, sc, u, start, warm):
    mu, su, my, sy = sc
    s0 = 0 if warm is None else max(0, start - warm)
    with torch.no_grad():
        un = torch.tensor((u[s0:] - mu) / su, dtype=torch.float64)[None, :, None]
        yh = net.double()(un)[0, :, 0].numpy() * sy + my
    net.float()
    return yh[start - s0:]


def train_lstm(case, cfg, hp, seed, data):
    ut, yt, nfit, sc = data["ut"], data["yt"], data["nfit"], data["sc"]
    mu, su, my, sy = sc
    T = min(cfg["T_lstm"], nfit // 2)
    wl = T // 4
    un = torch.tensor((ut[:nfit] - mu) / su, dtype=torch.float32)
    yn = torch.tensor((yt[:nfit] - my) / sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net = LSTMnet(hp["hidden"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    bs = 64
    best, bestS, bad = np.inf, None, 0
    for it in range(1, cfg["it_lstm"] + 1):
        st = rng.integers(0, nfit - T, bs)
        ii = torch.as_tensor(st[:, None] + np.arange(T)[None, :])
        yh = net(un[ii][:, :, None])[:, :, 0]
        loss = ((yh[:, wl:] - yn[ii][:, wl:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % cfg["ev"] == 0:
            e = nrmse(yt[nfit:], lstm_sim(net, sc, ut, nfit, cfg["warm"]))
            if e < best:
                best, bestS, bad = e, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if bestS is not None:
        net.load_state_dict(bestS)
    h = hp["hidden"]
    a = 4 * h * (1 + h) + h + h + h + 2      # LSTM (regra do E6) + saida linear + esc./desesc.
    m = 4 * h * (1 + h) + 3 * h + h + h + 2
    sim = lambda u, y, s: lstm_sim(net, sc, u, s, cfg["warm"])
    W = {k: v.double().numpy().tolist() for k, v in net.state_dict().items()}
    return best, it, sim, dict(A=a, M=m, act=f"{3 * h} sigm + {2 * h} tanh"), {"state": W}


# ------------------------------------------------------------------ SUBNET
class SUBNET(nn.Module):
    def __init__(self, lag, nx, width, depth):
        super().__init__()
        self.enc = MLPb(2 * lag, nx, width, depth, True)
        self.f = MLPb(nx + 1, nx, width, depth, True)
        self.h = MLPb(nx, 1, width, depth, True)


def subnet_sim(Pn, sc, u, y, start, lag):
    mu, su, my, sy = sc
    yn, un = (y - my) / sy, (u - mu) / su
    x = mlpb_eval(Pn["enc"], np.r_[yn[start - lag:start][::-1], un[start - lag:start][::-1]])
    out = np.empty(len(y) - start)
    for i, k in enumerate(range(start, len(y))):
        out[i] = mlpb_eval(Pn["h"], x)[0] * sy + my
        x = mlpb_eval(Pn["f"], np.r_[x, un[k]])
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
            return None
    return out


def train_subnet(case, cfg, hp, seed, data):
    ut, yt, nfit, sc = data["ut"], data["yt"], data["nfit"], data["sc"]
    mu, su, my, sy = sc
    lag, nx, T = cfg["lag"], cfg["nx"], cfg["T_sub"]
    un = torch.tensor((ut[:nfit] - mu) / su, dtype=torch.float32)
    yn = torch.tensor((yt[:nfit] - my) / sy, dtype=torch.float32)
    torch.manual_seed(seed)
    net = SUBNET(lag, nx, hp["width"], hp["depth"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    bs = 256 if nfit > 5000 else 64
    lags = np.arange(1, lag + 1)
    best, bestP, bad = np.inf, None, 0
    for it in range(1, cfg["it_sub"] + 1):
        st = rng.integers(lag, nfit - T, bs)
        past = torch.cat([yn[torch.as_tensor(st[:, None] - lags)], un[torch.as_tensor(st[:, None] - lags)]], 1)
        x = net.enc(past)
        loss = 0.0
        for t in range(T):
            k = torch.as_tensor(st + t)
            loss = loss + ((net.h(x)[:, 0] - yn[k]) ** 2).mean()
            x = net.f(torch.cat([x, un[k][:, None]], 1))
        loss = loss / T
        opt.zero_grad(); loss.backward(); opt.step()
        if it % cfg["ev"] == 0:
            Pn = {k: mlpb_numpy(getattr(net, k)) for k in ("enc", "f", "h")}
            e = nrmse(yt[nfit:], subnet_sim(Pn, sc, ut, yt, nfit, lag))
            if e < best:
                best, bestP, bad = e, Pn, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if bestP is None:
        bestP = {k: mlpb_numpy(getattr(net, k)) for k in ("enc", "f", "h")}
    af, mf, tf = mlpb_ops(nx + 1, nx, hp["width"], hp["depth"], True)
    ah, mh, th = mlpb_ops(nx, 1, hp["width"], hp["depth"], True)
    ae, me, te = mlpb_ops(2 * lag, nx, hp["width"], hp["depth"], True)
    sim = lambda u, y, s: subnet_sim(bestP, sc, u, y, s, lag)
    ops = dict(A=af + ah + 2, M=mf + mh + 2, act=f"{tf + th} tanh",
               encoder_once=dict(A=ae, M=me, act=f"{te} tanh"))
    return best, it, sim, ops, {"P": {k: jsonable(v) for k, v in bestP.items()}}


TRAIN = {"mlp": train_mlp, "lstm": train_lstm, "subnet": train_subnet}


def main():
    case, model = sys.argv[1], sys.argv[2]
    quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
    cfg = dict(CFG[case])
    if quick:
        for k in ("it_mlp", "it_lstm", "it_sub"):
            cfg[k] = 2 * cfg["ev"]
    ut, yt, uv, yv, start = load_case(case)
    nfit = int(round(0.8 * len(yt)))
    sc = (ut[:nfit].mean(), ut[:nfit].std(), yt[:nfit].mean(), yt[:nfit].std())
    data = dict(ut=ut, yt=yt, nfit=nfit, sc=sc)
    runs, keep = [], []
    for hp in GRID[model]:
        for seed in SEEDS:
            t0 = time.perf_counter()
            hold, nit, sim, ops, params = TRAIN[model](case, cfg, hp, seed, data)
            ttrain = time.perf_counter() - t0
            yh = sim(uv, yv, start)
            r = dict(hp=hp, seed=seed, train_s=ttrain, iterations=nit, holdout_nrmse=hold,
                     val_nrmse=nrmse(yv[start:], yh), **{k: v for k, v in ops.items()})
            r["C_karatsuba"] = round(r["A"] * 64 + r["M"] * 64 ** 1.585, 2)
            runs.append(r)
            keep.append((params, yh))
            print(json.dumps({k: v for k, v in r.items()}), flush=True)
    sel = int(np.argmin([r["holdout_nrmse"] for r in runs]))
    total = sum(r["train_s"] for r in runs)
    res = dict(case=case, model=model, quick=quick, threads=1, nfit=nfit, nholdout=len(yt) - nfit,
               scaling=dict(mu=sc[0], su=sc[1], my=sc[2], sy=sc[3]), cfg=cfg, runs=runs,
               selected=sel, selected_run=runs[sel],
               search_train_s=total, search_carbon_mg=total * RATE_MG_S,
               selected_train_s=runs[sel]["train_s"], selected_carbon_mg=runs[sel]["train_s"] * RATE_MG_S,
               weights=keep[sel][0],
               yhat_val=None if keep[sel][1] is None else keep[sel][1].tolist())
    name = f"{case}_{model}{'_quick' if quick else ''}.json"
    (OUT / name).write_text(json.dumps(res), encoding="utf-8")
    s = runs[sel]
    print(f"== {case} {model}: escolhido {s['hp']} seed {s['seed']} | holdout {s['holdout_nrmse']:.3f} "
          f"| validacao {s['val_nrmse']:.3f} | A/M {s['A']}/{s['M']} | busca {total:.1f} s "
          f"({total * RATE_MG_S:.0f} mg CO2e)", flush=True)


if __name__ == "__main__":
    main()
