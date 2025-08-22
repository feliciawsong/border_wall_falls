#--- border wall data analysis 1 ---

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- Load --- path to CSV
df = pd.read_csv("/Users/feliciasong/Desktop/dispo_fu.csv")

# Exclude anyone whose disposition was "EXPIRED"
if 'dispo' in df.columns:
    df = df[df['dispo'].astype(str).str.strip().str.upper() != 'EXPIRED']

# --- Basic cleaning ---
# Treat 0/1/NaN as 0/1/"Missing" for these binary-ish indicators
for col in ["multilingual_dispo", "family"]:
    if col in df.columns: df[col] = df[col].fillna("Missing").astype("category")

# Ensure presence & categorical typing for key categoricals
for col in ["ethnicity", "lang", "sex", "border", "operative", "discharge_death", "dispo"]:
    if col in df.columns:
        df[col] = df[col].fillna("Missing").astype("category")

# Numeric covariates
for col in ["age", "los"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Drop discharge date - don't want it as a predictor ---
if "discharge_death" in df.columns:
    df = df.drop(columns=["discharge_death"])

# ---  Drop ethnicity---
if "ethnicity" in df.columns:
    df = df.drop(columns=["ethnicity"])

# --- Collapse rare dispo categories into other ---
if "dispo" in df.columns:
    df["dispo"] = df["dispo"].astype(str).str.strip().str.upper().replace(
        {
            "ACUTE CARE FACILITY": "OTHER",
            "HOSP TRANSFER": "OTHER",
            "LTCH": "OTHER",
        }
    ).astype("category")

# --- Language cleanup & grouping ---
if "lang" in df.columns:
    df["lang_grouped"] = df["lang"].astype(str).str.strip().str.capitalize().apply(
        lambda x: "Spanish" if x == "Spanish"
        else "English" if x == "English"
        else "Other"
    ).astype("category")

# --- Border cleanup ---
if "border" in df.columns:
    df["border"] = df["border"].astype(str).str.strip().str.upper().astype("category")

# --- Outcome: ensure binary numeric ---
if "fu" not in df.columns:
    raise ValueError("Expected an 'fu' column (0/1) for the outcome.")
df["fu"] = pd.to_numeric(df["fu"], errors="coerce")
print("Outcome counts (pre-dropna):\n", df["fu"].value_counts(dropna=False), "\n")
df = df.dropna(subset=["fu"])
df["fu"] = df["fu"].astype(int)

# --- Build formula for the model ---
terms = [
    "C(multilingual_dispo)", "C(emmi)", "C(lsw)", "C(family)",
    "C(lang_grouped)", "age", "C(sex)",
    "C(operative)", "los", "C(dispo)"
]
# Filter for terms that have columns in the DataFrame
rhs = " + ".join([t for t in terms if t.replace("C(", "").replace(")", "") in df.columns])
formula = f"fu ~ {rhs}"
print("Model formula:", formula)

# --- Fit GLM with HC3 standard errors ---
model = smf.glm(formula, data=df, family=sm.families.Binomial())
res = model.fit(cov_type="HC3")
print(res.summary())

# --- Check for complete separation ---
def xtab(col, y="fu"):
    t = pd.crosstab(df[col], df[y], dropna=False)
    zero_cell_by_row = (t == 0).any(axis=1)
    print(f"\n{col} x {y}\n", t)
    if zero_cell_by_row.any():
        print("  -> levels with a zero cell (possible separation):", list(t.index[zero_cell_by_row]))

print("\nChecking for separation:")
for col in [
    "dispo", "operative", "lsw", "emmi",
    "multilingual_dispo", "family", "lang_grouped", "sex"
]:
    if col in df.columns:
        xtab(col)
