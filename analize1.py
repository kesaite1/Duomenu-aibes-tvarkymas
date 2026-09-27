import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("A24.csv")

MY_COLS = [
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index"
]

print("Duomenų dydis:", df.shape)
print("\nStulpeliai:")
print(df.columns.tolist())


print("\n--- DUOMENŲ TIPAI ---")
print(df[MY_COLS].dtypes)

print("\n--- PIRMOJI EILUTĖS ---")
print(df[MY_COLS].head())

print("\n--- UNIKALIŲ REIKŠMIŲ SKAIČIUS ---")
print(df[MY_COLS].nunique())


df["Steel_Plate_Thickness"] = (
    df["Steel_Plate_Thickness"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0]
    .astype(float)
)

df["Empty_Index"] = pd.to_numeric(
    df["Empty_Index"],
    errors="coerce"
)


print("\n--- TRŪKSTAMOS REIKŠMĖS ---")
missing = df[MY_COLS].isna().sum()
print(missing)

print("\nTrūkstamų reikšmių procentas:")
print((missing / len(df) * 100).round(2))

print("\n--- DUBLIKATAI ---")
print("Dublikatų skaičius:", df.duplicated().sum())

print("\n--- PAGRINDINĖ STATISTIKA ---")
print(df[MY_COLS].describe().T)


print("\n--- REIKŠMIŲ RIBŲ PATIKRA ---")

checks = {
    "Minimum_of_Luminosity < 0":
        (df["Minimum_of_Luminosity"] < 0).sum(),

    "Maximum_of_Luminosity > 255":
        (df["Maximum_of_Luminosity"] > 255).sum(),

    "Minimum > Maximum":
        (df["Minimum_of_Luminosity"] >
         df["Maximum_of_Luminosity"]).sum(),

    "Length_of_Conveyer <= 0":
        (df["Length_of_Conveyer"] <= 0).sum(),

    "Steel_Plate_Thickness <= 0":
        (df["Steel_Plate_Thickness"] <= 0).sum(),

    "Edges_Index outside [0,1]":
        ((df["Edges_Index"] < 0) |
         (df["Edges_Index"] > 1)).sum(),

    "Empty_Index outside [0,1]":
        ((df["Empty_Index"] < 0) |
         (df["Empty_Index"] > 1)).sum(),

    "Square_Index outside [0,1]":
        ((df["Square_Index"] < 0) |
         (df["Square_Index"] > 1)).sum(),

    "Outside_X_Index outside [0,1]":
        ((df["Outside_X_Index"] < 0) |
         (df["Outside_X_Index"] > 1)).sum()
}

for name, count in checks.items():
    print(f"{name}: {count}")


print("\n--- KLASIŲ PASISKIRSTYMAS ---")
print(df["class"].value_counts())

print("\nProcentais:")
print(
    (df["class"].value_counts(normalize=True) * 100)
    .round(2)
)

plt.figure(figsize=(8, 4))
df["class"].value_counts().plot(kind="bar")
plt.title("Objektų pasiskirstymas pagal klasę")
plt.xlabel("Klasė")
plt.ylabel("Objektų skaičius")
plt.tight_layout()
plt.show()


print("\n--- IŠSKIRTYS PAGAL IQR ---")

outlier_counts = {}

for col in MY_COLS:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_counts[col] = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

print(pd.Series(outlier_counts))


print("\n--- APRAŠOMOJI STATISTIKA ---")

stats = df[MY_COLS].describe().T
stats["median"] = df[MY_COLS].median()
stats["missing_%"] = (
    df[MY_COLS].isna().mean() * 100
).round(2)

print(stats)


for col in MY_COLS:
    plt.figure(figsize=(6, 4))
    df[col].hist(bins=20)
    plt.title(f"{col} pasiskirstymas")
    plt.xlabel(col)
    plt.ylabel("Dažnis")
    plt.tight_layout()
    plt.show()


for col in MY_COLS:
    plt.figure(figsize=(7, 4))
    df.boxplot(column=col, by="class")
    plt.title(f"{col} pagal klasę")
    plt.suptitle("")
    plt.xlabel("Klasė")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


correlation = df[MY_COLS].corr(method="spearman")

print("\n--- SPEARMAN KORELIACIJOS ---")
print(correlation.round(2))

plt.figure(figsize=(9, 7))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Spearman koreliacija")

plt.xticks(
    range(len(MY_COLS)),
    MY_COLS,
    rotation=90
)

plt.yticks(
    range(len(MY_COLS)),
    MY_COLS
)

plt.title("Požymių Spearman koreliacija")
plt.tight_layout()
plt.show()


cleaned = df.copy()

cleaned.loc[
    (cleaned["Edges_Index"] < 0) |
    (cleaned["Edges_Index"] > 1),
    "Edges_Index"
] = np.nan

print("\n--- TRŪKSTAMOS REIKŠMĖS PRIEŠ PARUOŠIMĄ ---")
print(cleaned[MY_COLS].isna().sum())


for col in MY_COLS:
    if cleaned[col].isna().sum() > 0:
        cleaned[col] = cleaned[col].fillna(
            cleaned[col].median()
        )

print("\n--- TRŪKSTAMOS REIKŠMĖS PO PARUOŠIMO ---")
print(cleaned[MY_COLS].isna().sum())


experiment = df.copy()

known = experiment[
    experiment["Edges_Index"].notna()
].copy()

np.random.seed(42)

test_index = known.sample(
    frac=0.20
).index

true_values = known.loc[
    test_index,
    "Edges_Index"
]

mean_prediction = known.loc[
    ~known.index.isin(test_index),
    "Edges_Index"
].mean()

median_prediction = known.loc[
    ~known.index.isin(test_index),
    "Edges_Index"
].median()

mean_pred = pd.Series(
    mean_prediction,
    index=test_index
)

median_pred = pd.Series(
    median_prediction,
    index=test_index
)


def calculate_metrics(true, predicted):
    mae = np.mean(np.abs(true - predicted))
    rmse = np.sqrt(np.mean((true - predicted) ** 2))
    return mae, rmse


mean_mae, mean_rmse = calculate_metrics(
    true_values,
    mean_pred
)

median_mae, median_rmse = calculate_metrics(
    true_values,
    median_pred
)

results = pd.DataFrame({
    "Metodas": ["Vidurkis", "Mediana"],
    "MAE": [mean_mae, median_mae],
    "RMSE": [mean_rmse, median_rmse]
})

print("\n--- PALYGINIMO EKSPERIMENTAS ---")
print(results)


print("\n--- DUOMENŲ POKYTIS ---")

print("Pradinis eilučių skaičius:", len(df))
print("Paruoštų eilučių skaičius:", len(cleaned))

print("\nTrūkstamos reikšmės prieš:")
print(df[MY_COLS].isna().sum().sum())

print("Trūkstamos reikšmės po:")
print(cleaned[MY_COLS].isna().sum().sum())


cleaned[
    MY_COLS + ["class"]
].to_csv(
    "A24_cleaned.csv",
    index=False
)

print("\nParuošti duomenys išsaugoti faile: A24_cleaned.csv")