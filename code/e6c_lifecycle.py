"""E6c - custo no ciclo de vida: identificacao/treino + N avaliacoes.

Junta polinomios (E3b tempo por amostra, E4 tempo de identificacao) e redes
(E6c treino com busca + tempo por amostra), todos medidos no mesmo PC com 1
thread, e converte tempo em CO2e com a taxa da Tabela 1 (121.81/60 mg/s).

Saidas: fase2_outputs/e6c/e6c_lifecycle.csv e e6c_lifecycle.md
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F2 = HERE / "fase2_outputs"
RATE = 121.81 / 60.0                              # mg CO2e por segundo (1 nucleo)
FS = {"dc": None, "ced": 50.0, "silverbox": 1e7 / 2 ** 14}   # Hz
YEAR = 365.25 * 24 * 3600


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    e3 = [r for r in read_csv(F2 / "e3b" / "e3b_timing.csv") if r["profile"] == "direto"]
    e4 = {r["case"]: r for r in read_csv(F2 / "e4" / "e4_ident_time.csv")}
    nn = [r for r in read_csv(F2 / "e6c" / "e6c_timing.csv") if r["profile"] == "direto"]
    rows = []
    for case in ["dc", "ced", "silverbox"]:
        d = json.loads((F2 / "e3b" / f"{case}_models.json").read_text(encoding="utf-8"))
        y = np.asarray(d["y"], float)[int(d["start0"]):]
        nr = {}
        for m in d["models"]:
            yh = np.array([np.nan if v is None else v for v in m["yhat"]], float)
            nr[m["label"]] = float(np.linalg.norm(y - yh) / np.linalg.norm(y - y.mean()))
        for r in e3:
            if r["case"] != case or r["label"] in ("Green (0.00)",):
                continue
            ident = float(e4[case]["tFROLS_s"] if r["label"] in ("SysIdentPy", "Green (1.00)") else e4[case]["tGreen_s"])
            rows.append(dict(case=case, family="polinomial", model=r["label"], A=int(r["A"]), M=int(r["M"]),
                             activations="", C=float(r["C"]), NRMSE=nr[r["label"]], ident_s=ident,
                             ns_sample=float(r["ns_median"])))
        for r in nn:
            if r["case"] != case:
                continue
            rows.append(dict(case=case, family="rede", model=r["model"].upper() if r["model"] != "subnet" else "SUBNET",
                             A=int(r["A"]), M=int(r["M"]), activations=r["activations"], C=float(r["C"]),
                             NRMSE=float(r["val_nrmse"]), ident_s=float(r["search_train_s"]),
                             ns_sample=float(r["ns_median"])))
    for r in rows:
        r["ident_mgCO2e"] = r["ident_s"] * RATE
        r["mg_per_1e9_samples"] = r["ns_sample"] * RATE          # 1e9 amostras x ns = s
        fs = FS[r["case"]]
        r["g_per_device_year"] = None if fs is None else r["ns_sample"] * 1e-9 * fs * YEAR * RATE / 1000
        # amostras ate o custo de avaliacao igualar o de identificacao/treino
        r["N_eval_equals_ident"] = r["ident_s"] / (r["ns_sample"] * 1e-9)
        r["days_eval_equals_ident"] = None if fs is None else r["N_eval_equals_ident"] / fs / 86400
    out = F2 / "e6c" / "e6c_lifecycle.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["| Caso | Modelo | A/M | NRMSE | Identificação/treino (s) | CO₂e ident. (mg) | ns/amostra | CO₂e por 10⁹ amostras (mg) | g CO₂e por dispositivo-ano | Dias até avaliação = treino |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        gy = "" if r["g_per_device_year"] is None else f"{r['g_per_device_year']:.3g}"
        dd = "" if r["days_eval_equals_ident"] is None else f"{r['days_eval_equals_ident']:.3g}"
        L.append(f"| {r['case']} | {r['model']} | {r['A']}/{r['M']} | {r['NRMSE']:.3f} | {r['ident_s']:.3g} | "
                 f"{r['ident_mgCO2e']:.3g} | {r['ns_sample']:.3g} | {r['mg_per_1e9_samples']:.3g} | {gy} | {dd} |")
    (F2 / "e6c" / "e6c_lifecycle.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
