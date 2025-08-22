import pandas as pd

# Load data
df = pd.read_csv("/Users/feliciasong/Desktop/border_data/Spine_fx_2021.csv", encoding='latin1')

# Define total cohort size
total_patients = 127

# Follow-up
df['Follow_Up'] = pd.to_numeric(df['Follow_Up'], errors='coerce')
follow_up_count = (df['Follow_Up'] == 1).sum()
percent_follow_up = (follow_up_count / total_patients) * 100
print(f"{follow_up_count} out of {total_patients} patients ({percent_follow_up:.2f}%) had a follow-up visit.")

# Ethnicity
not_h_count = (df['Ethnicity'] != 'H').sum()
percent_not_h = (not_h_count / total_patients) * 100
print(f"{not_h_count} out of {total_patients} patients ({percent_not_h:.2f}%) were not labeled as 'H' in the Ethnicity column.")

# Language
df['Language_clean'] = df['Language'].astype(str).str.strip().str.lower()
non_en_es_count = (~df['Language_clean'].isin(['english', 'spanish'])).sum()
percent_other_languages = (non_en_es_count / total_patients) * 100
print(f"{non_en_es_count} out of {total_patients} patients ({percent_other_languages:.2f}%) spoke a language other than English or Spanish.")

# Multilingual discharge resources
df['multilingual dispo'] = pd.to_numeric(df['multilingual dispo'], errors='coerce')
multilingual_count = (df['multilingual dispo'] == 1).sum()
percent_multilingual = (multilingual_count / total_patients) * 100
print(f"{multilingual_count} out of {total_patients} patients ({percent_multilingual:.2f}%) received multilingual discharge resources.")

# EMMI
df['EMMI'] = pd.to_numeric(df['EMMI'], errors='coerce')
emmi_count = (df['EMMI'] == 1).sum()
percent_emmi = (emmi_count / total_patients) * 100
print(f"{emmi_count} out of {total_patients} patients ({percent_emmi:.2f}%) received EMMI.")

# Charity resources
df['charity'] = pd.to_numeric(df['charity'], errors='coerce')
charity_count = (df['charity'] == 1).sum()
percent_charity = (charity_count / total_patients) * 100
print(f"{charity_count} out of {total_patients} patients ({percent_charity:.2f}%) received charity resources.")

# Social Work consult
df['LSW'] = pd.to_numeric(df['LSW'], errors='coerce')
lsw_count = (df['LSW'] == 1).sum()
percent_lsw = (lsw_count / total_patients) * 100
print(f"{lsw_count} out of {total_patients} patients ({percent_lsw:.2f}%) had a social work consult.")
import pandas as pd
from scipy.stats import chi2_contingency

# Load your data
df = pd.read_csv("/Users/feliciasong/Desktop/border_data/Spine_fx_2021.csv", encoding='latin1')

# Ensure numeric format
df['Follow_Up'] = pd.to_numeric(df['Follow_Up'], errors='coerce')
df['multilingual dispo'] = pd.to_numeric(df['multilingual dispo'], errors='coerce')

# Drop rows with missing data in either variable
subset_df = df[['Follow_Up', 'multilingual dispo']].dropna()

# Create contingency table
contingency = pd.crosstab(subset_df['multilingual dispo'], subset_df['Follow_Up'])

# Run chi-square test
chi2, p, dof, expected = chi2_contingency(contingency)

# Print output
print("Contingency Table:\n", contingency)
print(f"\nChi-square test statistic = {chi2:.4f}")
print(f"p-value = {p:.4f}")

if p < 0.05:
    print("There is a statistically significant association between multilingual discharge and follow-up.")
else:
    print("There is no statistically significant association between multilingual discharge and follow-up.")
import pandas as pd
from scipy.stats import chi2_contingency

# Load data
df = pd.read_csv("/Users/feliciasong/Desktop/border_data/Spine_fx_2021.csv", encoding='latin1')

# Ensure relevant columns are numeric
cols_to_check = ['Follow_Up', 'multilingual dispo', 'EMMI', 'charity', 'LSW']
for col in cols_to_check:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define function to run chi-square test
def run_chi_square(df, var1, var2):
    subset_df = df[[var1, var2]].dropna()
    contingency = pd.crosstab(subset_df[var1], subset_df[var2])
    chi2, p, dof, expected = chi2_contingency(contingency)

    print(f"\n--- {var2} vs. {var1} ---")
    print("Contingency Table:\n", contingency)
    print(f"Chi-square = {chi2:.4f}, p = {p:.4f}")

    if p < 0.05:
        print("→ Statistically significant association.")
    else:
        print("→ No statistically significant association.")

# Run tests with Follow_Up as outcome
run_chi_square(df, 'Follow_Up', 'multilingual dispo')
run_chi_square(df, 'Follow_Up', 'EMMI')
run_chi_square(df, 'Follow_Up', 'charity')
run_chi_square(df, 'Follow_Up', 'LSW')

import pandas as pd
from scipy.stats import fisher_exact

# Load data
df = pd.read_csv("/Users/feliciasong/Desktop/border_data/Spine_fx_2021.csv", encoding='latin1')

# Ensure columns are numeric
cols = ['Follow_Up', 'multilingual dispo', 'EMMI', 'charity', 'LSW']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Function to run Fisher's Exact Test
def run_fisher_test(df, exposure, outcome='Follow_Up'):
    subset = df[[exposure, outcome]].dropna()
    table = pd.crosstab(subset[exposure], subset[outcome])

    # Ensure it's a 2x2 table
    if table.shape == (2, 2):
        oddsratio, p_value = fisher_exact(table)
        print(f"\n--- {exposure} vs. {outcome} ---")
        print("Contingency Table:\n", table)
        print(f"Fisher's Exact Test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("→ Statistically significant association.")
        else:
            print("→ No statistically significant association.")
    else:
        print(f"\n{exposure} vs. {outcome} is not a 2x2 table, skipping.")

# Run Fisher tests
run_fisher_test(df, 'multilingual dispo')
run_fisher_test(df, 'EMMI')
run_fisher_test(df, 'charity')
run_fisher_test(df, 'LSW')

import pandas as pd
import statsmodels.api as sm

# Load and clean your data
df = pd.read_csv("/Users/feliciasong/Desktop/border_data/Spine_fx_2021.csv", encoding='latin1')

# Ensure variables are numeric
cols = ['Follow_Up', 'multilingual dispo', 'EMMI', 'charity', 'LSW']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in your selected variables
df_clean = df[cols].dropna()

# Define outcome and predictors
X = df_clean[['multilingual dispo', 'EMMI', 'charity', 'LSW']]
y = df_clean['Follow_Up']

# Add constant term (intercept) to the model
X = sm.add_constant(X)

# Fit logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Print summary
print(result.summary())
