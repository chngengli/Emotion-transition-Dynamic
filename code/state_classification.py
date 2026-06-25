import pandas as pd

file_path = r"E:\Analysis\EMA\usedata_cleaned.xlsx"
df = pd.read_excel(file_path)

# -------- type 1 --------
def discretize_fixed(x):
    if x <= 33:
        return "Low"
    elif x <= 66:
        return "Medium"
    else:
        return "High"

df["PA_state_fixed"] = df["PAmean"].apply(discretize_fixed)
df["NA_state_fixed"] = df["NAmean"].apply(discretize_fixed)

# -------- type2 --------
pa_q1, pa_q2 = df["PAmean"].quantile([0.33, 0.66])
na_q1, na_q2 = df["NAmean"].quantile([0.33, 0.66])

def discretize_quantile(x, q1, q2):
    if x <= q1:
        return "Low"
    elif x <= q2:
        return "Medium"
    else:
        return "High"

df["PA_state_tertile"] = df["PAmean"].apply(lambda x: discretize_quantile(x, pa_q1, pa_q2))
df["NA_state_tertile"] = df["NAmean"].apply(lambda x: discretize_quantile(x, na_q1, na_q2))

# -------- save --------
output_file_fixed = r"E:\Analysis\EMA\data_with_states_fixed.xlsx"
output_file_tertile = r"E:\Analysis\EMA\data_with_states_tertile.xlsx"

df.to_excel(output_file_fixed, index=False)
df.to_excel(output_file_tertile, index=False)

print(output_file_fixed)
print(output_file_tertile)

