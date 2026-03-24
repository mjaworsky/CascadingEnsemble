import pandas as pd
import numpy as np
from scipy.stats import chisquare

# --------------------------------------------------
# Table data
# --------------------------------------------------
data = [
("Alcohol","DT",0.013,0.015),
("Alcohol","RF",0.013,0.015),
("Alcohol","XG",0.011,0.001),

("Diabetes","DT",0.079,0.083),
("Diabetes","RF",0.087,0.088),
("Diabetes","XG",0.066,0.065),

("Gender+Injury+Fatigue","DT",0.079,0.084),
("Gender+Injury+Fatigue","RF",0.083,0.088),
("Gender+Injury+Fatigue","XG",0.066,0.065),

("Gender+Weight","DT",0.081,0.082),
("Gender+Weight","RF",0.081,0.084),
("Gender+Weight","XG",0.066,0.065),

("Smoking","DT",0.015,0.011),
("Smoking","RF",0.009,0.012),
("Smoking","XG",0.012,0.012),

("All Risk Factors","DT",0.097,0.103),
("All Risk Factors","RF",0.097,0.100),
("All Risk Factors","XG",0.093,0.091)
]

df = pd.DataFrame(data, columns=["RiskFactor","Algorithm","BaseF1","CascadeF1"])

# --------------------------------------------------
# Determine improvement
# --------------------------------------------------
df["Improved"] = df["CascadeF1"] > df["BaseF1"]

results = []

for alg in df["Algorithm"].unique():

    subset = df[df["Algorithm"] == alg]

    improved = subset["Improved"].sum()
    not_improved = len(subset) - improved

    observed = [improved, not_improved]

    # expected assuming no effect (50/50)
    expected = [len(subset)/2, len(subset)/2]

    chi2, p = chisquare(f_obs=observed, f_exp=expected)

    results.append({
        "Algorithm": alg,
        "Improved": improved,
        "NotImproved": not_improved,
        "Chi2": chi2,
        "p_value": p
    })

results_df = pd.DataFrame(results)

print(results_df)