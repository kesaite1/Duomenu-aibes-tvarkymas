
"""
1 laboratorinis darbas - mano dalis: stulpeliai I-P
(Minimum_of_Luminosity ... Outside_X_Index)
 
Paleidimas (terminale, su aktyvuota .venv):   python analize.py
Failas analize.py turi būti šalia A24.csv. Originalus A24.csv niekada nekeičiamas.
Rezultatai: outputs/tables (lentelės), outputs/figures (grafikai), data/processed (sutvarkyti duomenys)
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
SHOW = False          # True -> grafikai papildomai rodomi ekrane (kiekvienas langas stabdo kodą)
 
# ---------------------------------------------------------------------------
# 0. NUSTATYMAI: keliai, stulpeliai, grafikų stilius
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
RAW_FILE = next(p for p in (BASE / "A24.csv", BASE / "data" / "raw" / "A24.csv") if p.exists())
TAB_DIR, FIG_DIR, PROC_DIR = BASE / "outputs" / "tables", BASE / "outputs" / "figures", BASE / "data" / "processed"
for folder in (TAB_DIR, FIG_DIR, PROC_DIR):
    folder.mkdir(parents=True, exist_ok=True)
 
MY_COLS = ["Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer", "Steel_Plate_Thickness",
           "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index"]
 
BLUE, RED, TEAL, GREY = "#3B6FB6", "#D1495B", "#2A9D8F", "#6B7280"
CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"]
 
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA", "axes.edgecolor": "#CCCCCC",
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": "#374151",
    "xtick.color": "#374151", "ytick.color": "#374151", "font.size": 10,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
 
 
def save(fig, name):
    """Įrašo grafiką į outputs/figures (ir, jei SHOW=True, parodo ekrane)."""
    fig.savefig(FIG_DIR / name)
    if SHOW:
        plt.show()
    plt.close(fig)
 
 
def title(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)
 
 
# ---------------------------------------------------------------------------
# 1. DUOMENŲ ĮKĖLIMAS IR TIPAI
# ---------------------------------------------------------------------------
title("1. Duomenų įkėlimas")
raw = pd.read_csv(RAW_FILE)           # originalas - jo nekeičiame
print("Eilučių ir stulpelių skaičius:", raw.shape)
types_table = raw[MY_COLS].dtypes.astype(str).rename("Duomenų tipas")
print(types_table)
types_table.to_csv(TAB_DIR / "lentele_duomenu_tipai.csv")
 
# ---------------------------------------------------------------------------
# 2. TEKSTAS -> SKAIČIAI (dirbame su kopija 'df')
# ---------------------------------------------------------------------------
title("2. Tekstinių stulpelių konvertavimas")
df = raw.copy()
 
# Steel_Plate_Thickness: "40 mm" -> 40
assert df["Steel_Plate_Thickness"].str.endswith(" mm").all()
df["Steel_Plate_Thickness"] = df["Steel_Plate_Thickness"].str.replace(" mm", "", regex=False).astype(int)
 
# Empty_Index: žodžiai ("error", "unknown", "?", "not_measured") -> NaN
as_number = pd.to_numeric(raw["Empty_Index"], errors="coerce")
print("Ne skaitinės Empty_Index reikšmės:\n", raw.loc[as_number.isna(), "Empty_Index"].value_counts())
df["Empty_Index"] = as_number
 
# ---------------------------------------------------------------------------
# 3. MIN/MAX IR TRŪKSTAMOS REIKŠMĖS
# ---------------------------------------------------------------------------
title("3. Min/maks ir trūkstamos reikšmės")
minmax = df[MY_COLS].agg(["min", "max"]).T
print(minmax)
minmax.to_csv(TAB_DIR / "lentele_min_max.csv")
 
missing = pd.DataFrame({"Trūksta": df[MY_COLS].isna().sum(), "Proc.": (df[MY_COLS].isna().mean() * 100).round(2)})
print(missing)
missing.to_csv(TAB_DIR / "lentele_truksta.csv")
 
# ---------------------------------------------------------------------------
# 4. LOGINIŲ RIBŲ PATIKRA
# ---------------------------------------------------------------------------
title("4. Loginių ribų patikra")
rules = {
    "Šviesumas ne intervale [0; 255]":
        (df.Minimum_of_Luminosity < 0) | (df.Minimum_of_Luminosity > 255) |
        (df.Maximum_of_Luminosity < 0) | (df.Maximum_of_Luminosity > 255),
    "Minimum_of_Luminosity > Maximum_of_Luminosity": df.Minimum_of_Luminosity > df.Maximum_of_Luminosity,
    "Length_of_Conveyer <= 0 arba Steel_Plate_Thickness <= 0":
        (df.Length_of_Conveyer <= 0) | (df.Steel_Plate_Thickness <= 0),
    "Edges_Index < 0 arba > 1": (df.Edges_Index < 0) | (df.Edges_Index > 1),
    "Empty_Index < 0 arba > 1": (df.Empty_Index < 0) | (df.Empty_Index > 1),
    "Square_Index <= 0 arba > 1": (df.Square_Index <= 0) | (df.Square_Index > 1),
    "Outside_X_Index <= 0 arba > 1": (df.Outside_X_Index <= 0) | (df.Outside_X_Index > 1),
}
bounds = pd.Series({name: int(mask.sum()) for name, mask in rules.items()}, name="Pažeidimų skaičius")
print(bounds)
bounds.to_csv(TAB_DIR / "lentele_loginiu_ribu_patikra.csv")
 
# ---------------------------------------------------------------------------
# 5. KRYŽMINĖ PATIKRA: indeksus galima perskaičiuoti iš kitų stulpelių
# ---------------------------------------------------------------------------
title("5. Kryžminė patikra su formulėmis")
dx = df.X_Maximum - df.X_Minimum          # defekto plotis X kryptimi
dy = df.Y_Maximum - df.Y_Minimum          # defekto aukštis Y kryptimi
L = df.Length_of_Conveyer
box = dx * dy                             # apgaubiančio stačiakampio plotas
 
edges_calc = np.minimum(df.X_Minimum, L - df.X_Maximum) / (L / 2)
empty_calc = 1 - df.Pixels_Areas / box
square_calc = np.minimum(dx, dy) / np.maximum(dx, dy)
outx_calc = dx / L
 
valid_x = dx > 0                                                        # X koordinatės tvarkingos
valid_box = valid_x & (dy > 0) & (df.Pixels_Areas > 0) & (df.Pixels_Areas <= box)
 
 
def compare(name, given, calc, valid):
    ok = valid & given.notna()
    diff = (given - calc)[ok].abs()
    return {"Požymis": name, "Patikrinta eilučių": int(ok.sum()),
            "Nesutampa (>0.001)": int((diff > 0.001).sum()), "Didžiausias skirtumas": round(float(diff.max()), 5)}
 
 
cross = pd.DataFrame([
    compare("Edges_Index", df.Edges_Index.where(df.Edges_Index <= 1), edges_calc, valid_x),
    compare("Empty_Index", df.Empty_Index, empty_calc, valid_box),
    compare("Square_Index", df.Square_Index, square_calc, valid_x),
    compare("Outside_X_Index", df.Outside_X_Index, outx_calc, valid_x),
])
print(cross.to_string(index=False))
cross.to_csv(TAB_DIR / "lentele_kryzmine_patikra.csv", index=False)
 
# ---------------------------------------------------------------------------
# 6. IŠSKIRTYS (IQR) - tik nustatome, nešaliname
# ---------------------------------------------------------------------------
title("6. Išskirtys pagal IQR")
rows = {}
for col in MY_COLS:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n = int(((s < low) | (s > high)).sum())
    rows[col] = {"Q1": q1, "Q3": q3, "Apatinė riba": low, "Viršutinė riba": high,
                 "Išskirčių sk.": n, "Proc.": round(100 * n / len(s), 1)}
outliers = pd.DataFrame(rows).T.round(3)
print(outliers)
outliers.to_csv(TAB_DIR / "lentele_isskirtys_iqr.csv")
 
# ---------------------------------------------------------------------------
# 7. APRAŠOMOJI STATISTIKA
# ---------------------------------------------------------------------------
title("7. Aprašomoji statistika")
stats = df[MY_COLS].describe().T
stats["skewness"] = df[MY_COLS].skew()
stats["dispersija"] = df[MY_COLS].var()
print(stats.round(4))
stats.round(4).to_csv(TAB_DIR / "lentele_aprasomoji_statistika_pries.csv")
 
by_class = df.groupby("class")[MY_COLS].median().round(3)
print("\nMedianos pagal klasę:\n", by_class)
by_class.to_csv(TAB_DIR / "lentele_medianos_pagal_klase.csv")
 
# ---------------------------------------------------------------------------
# 8. GRAFIKAI
# ---------------------------------------------------------------------------
title("8. Grafikai")
classes = sorted(df["class"].unique())
color_of = dict(zip(classes, CLASS_COLORS))
 
# 8a) Klasių pasiskirstymas
counts = df["class"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(counts.index, counts.values, color=[color_of[c] for c in counts.index])
ax.bar_label(bars, padding=3)
ax.set_title("Objektų pasiskirstymas pagal defekto klasę")
ax.set_xlabel("Objektų skaičius")
ax.grid(axis="y", visible=False)
save(fig, "klasiu_pasiskirstymas.png")
 
# 8b) Histogramos su mediana
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), MY_COLS):
    s = df[col].dropna()
    ax.hist(s, bins=30, color=BLUE, edgecolor="white", alpha=0.9)
    ax.axvline(s.median(), color=RED, ls="--", lw=1.6, label=f"mediana = {s.median():.4g}")
    ax.set_title(col)
    ax.set_xlabel("Reikšmė")
    ax.set_ylabel("Dažnis")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle("Požymių pasiskirstymas (prieš valymą)", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "histogramos_pries.png")
 
# 8c) Stačiakampės diagramos pagal klasę
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), MY_COLS):
    data = [df.loc[df["class"] == c, col].dropna() for c in classes]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4, markeredgecolor=GREY))
    for patch, c in zip(bp["boxes"], classes):
        patch.set_facecolor(color_of[c])
        patch.set_alpha(0.85)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_title(col)
fig.suptitle("Požymių reikšmės pagal defekto klasę", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "boxplot_pagal_klase.png")
 
# 8d) Spearman koreliacijos
corr = df[MY_COLS].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(MY_COLS)), MY_COLS, rotation=45, ha="right")
ax.set_yticks(range(len(MY_COLS)), MY_COLS)
for i in range(len(MY_COLS)):
    for j in range(len(MY_COLS)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > 0.6 else "#111827")
ax.grid(False)
fig.colorbar(im, ax=ax, label="Spearman koreliacija", shrink=0.85)
ax.set_title("Požymių Spearman koreliacijos")
fig.tight_layout()
save(fig, "koreliacijos.png")
corr.round(3).to_csv(TAB_DIR / "lentele_koreliacijos_spearman.csv")
print("Grafikai įrašyti į", FIG_DIR)
 
# ---------------------------------------------------------------------------
# 9. VALYMAS (eilučių nešaliname - tik taisome reikšmes)
# ---------------------------------------------------------------------------
title("9. Duomenų valymas")
clean = df.copy()
log = {}
 
too_big = clean.Edges_Index > 1                                  # neįmanomos reikšmės -> NaN
log["Edges_Index > 1 pakeista į NaN"] = int(too_big.sum())
clean.loc[too_big, "Edges_Index"] = np.nan
 
miss = clean.Edges_Index.isna() & valid_x                        # užpildome pagal formulę
clean.loc[miss, "Edges_Index"] = edges_calc[miss].round(4)
log["Edges_Index užpildyta pagal formulę"] = int(miss.sum())
 
miss = clean.Empty_Index.isna() & valid_box
clean.loc[miss, "Empty_Index"] = empty_calc[miss].round(4)
log["Empty_Index užpildyta pagal formulę"] = int(miss.sum())
 
left = clean.Empty_Index.isna()                                  # jei formulė netinka - mediana
clean.loc[left, "Empty_Index"] = clean.Empty_Index.median()
log["Empty_Index užpildyta mediana"] = int(left.sum())
 
print(pd.Series(log))
print("Likę trūkstamų reikšmių:", int(clean[MY_COLS].isna().sum().sum()))
 
stats_after = clean[MY_COLS].describe().T
stats_after["skewness"] = clean[MY_COLS].skew()
stats_after["dispersija"] = clean[MY_COLS].var()
stats_after.round(4).to_csv(TAB_DIR / "lentele_aprasomoji_statistika_po.csv")
 
out = clean[MY_COLS + ["class"]].copy()
out.index.name = "row_id"                                        # sujungimui su kolegų dalimis
out.to_csv(PROC_DIR / "A24_I_P_clean.csv")
 
# Prieš / po grafikas
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, col in enumerate(["Edges_Index", "Empty_Index"]):
    bins = np.linspace(0, df[col].max(), 30)
    for k, (data, label, color) in enumerate([(df, "prieš valymą", BLUE), (clean, "po valymo", TEAL)]):
        ax = axes[row, k]
        ax.hist(data[col].dropna(), bins=bins, color=color, edgecolor="white")
        ax.set_title(f"{col} - {label}")
        ax.set_xlabel("Reikšmė")
        ax.set_ylabel("Dažnis")
        if col == "Edges_Index":
            ax.axvline(1, color=RED, ls="--", lw=1.2, label="viršutinė riba = 1")
            ax.legend(frameon=False, fontsize=8)
fig.suptitle("Apdorojimo poveikis požymių pasiskirstymui", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "pries_po.png")
 
# ---------------------------------------------------------------------------
# 10. PALYGINAMASIS EKSPERIMENTAS: Edges_Index užpildymo būdai
#     Žinomų reikšmių dalį (20 %) "paslepiame", užpildome trim būdais ir lyginame su tiesa.
# ---------------------------------------------------------------------------
title("10. Palyginamasis eksperimentas (Edges_Index)")
known = df.Edges_Index.notna() & (df.Edges_Index <= 1) & valid_x
truth = df.Edges_Index[known]
 
rng = np.random.default_rng(42)                                  # fiksuotas seed -> atkuriamumas
hidden = rng.choice(truth.index, size=int(0.2 * len(truth)), replace=False)
visible = truth.drop(hidden)
 
methods = {
    "Vidurkis": pd.Series(visible.mean(), index=hidden),
    "Mediana": pd.Series(visible.median(), index=hidden),
    "Formulė": edges_calc[hidden].round(4),
}
 
ref_col = df.Minimum_of_Luminosity[truth.index]
rows = [{"Metodas": "Tikra (etalonas)", "MAE": 0.0, "RMSE": 0.0, "Std po užpildymo": round(truth.std(), 4),
         "Corr su Min_Luminosity": round(truth.corr(ref_col), 4)}]
for name, pred in methods.items():
    err = pred - truth[hidden]
    full = pd.concat([visible, pred]).loc[truth.index]
    rows.append({"Metodas": name, "MAE": round(err.abs().mean(), 4), "RMSE": round(np.sqrt((err ** 2).mean()), 4),
                 "Std po užpildymo": round(full.std(), 4), "Corr su Min_Luminosity": round(full.corr(ref_col), 4)})
experiment = pd.DataFrame(rows)
print(f"Paslėpta reikšmių: {len(hidden)} iš {len(truth)}")
print(experiment.to_string(index=False))
experiment.to_csv(TAB_DIR / "lentele_eksperimentas_edges_index.csv", index=False)
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, (name, pred), color in zip(axes, methods.items(), [GREY, GREY, TEAL]):
    rmse = np.sqrt(((pred - truth[hidden]) ** 2).mean())
    ax.plot([0, 1], [0, 1], color=RED, lw=1.2, label="tobulas atitikimas")
    ax.scatter(truth[hidden], pred, s=14, alpha=0.6, color=color, edgecolor="none")
    ax.set_title(f"{name}  (RMSE = {rmse:.3f})")
    ax.set_xlabel("Tikroji reikšmė")
    ax.set_aspect("equal")
axes[0].set_ylabel("Užpildyta reikšmė")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Edges_Index: užpildytos ir tikrosios reikšmės", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "eksperimentas_edges_index.png")
 
print("\nBaigta. Lentelės:", TAB_DIR, "| grafikai:", FIG_DIR)
 
