# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"E:\Analysis\EMA\extra")

INPUT_FILE = BASE_DIR / "usedata_cleaned.xlsx"

OUTPUT_FILE = (
    BASE_DIR / "data_with_all_state_classifications.xlsx"
)

FIG_DIR = (
    BASE_DIR / "state_classification_figures"
)


ID_COL = "ID"

PA_COL = "PAmean"
NA_COL = "NAmean"

LEVELS = ["Low", "Medium", "High"]

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Load data
# =========================
df = pd.read_excel(INPUT_FILE)


# =========================
# General functions
# =========================
def as_ordered_category(values):
    """Convert Low, Medium, and High labels to an ordered categorical series."""

    return pd.Series(
        pd.Categorical(
            values,
            categories=LEVELS,
            ordered=True
        ),
        index=values.index
    )


def classify_by_cutoffs(x, q1, q2):
    """Classify values into Low, Medium, and High using two cutoffs."""

    out = pd.Series(
        pd.NA,
        index=x.index,
        dtype="object"
    )

    valid = x.notna().copy()

    if isinstance(q1, pd.Series):
        valid &= q1.notna()
    else:
        valid &= pd.notna(q1)

    if isinstance(q2, pd.Series):
        valid &= q2.notna()
    else:
        valid &= pd.notna(q2)

    out.loc[
        valid & (x <= q1)
    ] = "Low"

    out.loc[
        valid
        & (x > q1)
        & (x <= q2)
    ] = "Medium"

    out.loc[
        valid & (x > q2)
    ] = "High"

    return as_ordered_category(out)


def add_person_tertiles(
    data,
    id_col,
    value_col,
    emotion_prefix
):
    """Add person-specific tertile classifications and a strict three-level version."""

    stats = (
        data
        .groupby(
            id_col,
            dropna=False
        )[value_col]
        .agg(
            n_valid=lambda s: s.notna().sum(),

            n_unique=lambda s: (
                s.nunique(dropna=True)
            ),

            q33=lambda s: (
                s.quantile(1 / 3)
                if s.notna().any()
                else np.nan
            ),

            q67=lambda s: (
                s.quantile(2 / 3)
                if s.notna().any()
                else np.nan
            )
        )
        .reset_index()
    )

    stats = stats.rename(
        columns={
            "n_valid": (
                f"{emotion_prefix}_person_n_valid"
            ),
            "n_unique": (
                f"{emotion_prefix}_person_n_unique"
            ),
            "q33": (
                f"{emotion_prefix}_person_q33"
            ),
            "q67": (
                f"{emotion_prefix}_person_q67"
            )
        }
    )

    q33_col = (
        f"{emotion_prefix}_person_q33"
    )

    q67_col = (
        f"{emotion_prefix}_person_q67"
    )

    state_col = (
        f"{emotion_prefix}"
        "_state_person_tertile"
    )

    data = data.merge(
        stats,
        on=id_col,
        how="left"
    )

    data[state_col] = classify_by_cutoffs(
        data[value_col],
        data[q33_col],
        data[q67_col]
    )

    n_levels_col = (
        f"{emotion_prefix}"
        "_person_n_state_levels"
    )

    n_levels = (
        data
        .groupby(
            id_col,
            dropna=False
        )[state_col]
        .nunique(dropna=True)
        .rename(n_levels_col)
        .reset_index()
    )

    data = data.merge(
        n_levels,
        on=id_col,
        how="left"
    )

    stats = stats.merge(
        n_levels,
        on=id_col,
        how="left"
    )

    valid_col = (
        f"{emotion_prefix}"
        "_person_3level_valid"
    )

    stats[valid_col] = (
        (
            stats[
                f"{emotion_prefix}_person_n_valid"
            ] >= 3
        )
        & (
            stats[
                f"{emotion_prefix}_person_n_unique"
            ] >= 3
        )
        & (
            stats[q33_col]
            < stats[q67_col]
        )
        & (
            stats[n_levels_col] == 3
        )
    )

    data = data.merge(
        stats[
            [
                id_col,
                valid_col
            ]
        ],
        on=id_col,
        how="left"
    )

    strict_col = (
        f"{emotion_prefix}"
        "_state_person_tertile_strict"
    )

    strict_values = (
        data[state_col]
        .astype("object")
        .where(
            data[valid_col].fillna(False),
            pd.NA
        )
    )

    data[strict_col] = (
        as_ordered_category(strict_values)
    )

    return data, stats


def combine_pa_na(
    pa_state,
    na_state
):
    """Combine PA and NA states into nine joint affective states."""

    out = pd.Series(
        pd.NA,
        index=pa_state.index,
        dtype="object"
    )

    valid = (
        pa_state.notna()
        & na_state.notna()
    )

    out.loc[valid] = (
        "PA_"
        + pa_state.astype("string").loc[valid]
        + "__NA_"
        + na_state.astype("string").loc[valid]
    )

    return out


def linear_weighted_kappa(
    s1,
    s2
):
    """Compute linear weighted kappa for ordered three-level classifications."""

    tmp = pd.DataFrame(
        {
            "a": s1,
            "b": s2
        }
    ).dropna()

    if tmp.empty:
        return np.nan

    a = pd.Categorical(
        tmp["a"],
        categories=LEVELS,
        ordered=True
    ).codes

    b = pd.Categorical(
        tmp["b"],
        categories=LEVELS,
        ordered=True
    ).codes

    k = len(LEVELS)

    observed = np.zeros(
        (k, k),
        dtype=float
    )

    np.add.at(
        observed,
        (a, b),
        1
    )

    observed /= observed.sum()

    expected = np.outer(
        observed.sum(axis=1),
        observed.sum(axis=0)
    )

    disagreement = (
        np.abs(
            np.arange(k)[:, None]
            - np.arange(k)[None, :]
        )
        / (k - 1)
    )

    denominator = (
        disagreement
        * expected
    ).sum()

    if denominator == 0:
        return np.nan

    numerator = (
        disagreement
        * observed
    ).sum()

    return 1 - (
        numerator / denominator
    )


def confusion_tables(
    s1,
    s2
):
    """Return count and row-percentage confusion matrices."""

    tmp = pd.DataFrame(
        {
            "Method_A": s1,
            "Method_B": s2
        }
    ).dropna()

    counts = (
        pd.crosstab(
            tmp["Method_A"],
            tmp["Method_B"]
        )
        .reindex(
            index=LEVELS,
            columns=LEVELS,
            fill_value=0
        )
        .astype(int)
    )

    row_pct = (
        counts.div(
            counts
            .sum(axis=1)
            .replace(0, np.nan),
            axis=0
        )
        * 100
    )

    return counts, row_pct


# =========================
# Fixed absolute thresholds
# =========================
df["PA_state_fixed"] = (
    classify_by_cutoffs(
        df[PA_COL],
        33,
        66
    )
)

df["NA_state_fixed"] = (
    classify_by_cutoffs(
        df[NA_COL],
        33,
        66
    )
)


# =========================
# Group-level tertiles
# =========================
group_cutoff_rows = []

for emotion, value_col in [
    ("PA", PA_COL),
    ("NA", NA_COL)
]:

    q33 = df[value_col].quantile(1 / 3)
    q67 = df[value_col].quantile(2 / 3)

    df[
        f"{emotion}_state_group_tertile"
    ] = classify_by_cutoffs(
        df[value_col],
        q33,
        q67
    )

    group_cutoff_rows.append(
        {
            "emotion": emotion,
            "variable": value_col,
            "q33": q33,
            "q67": q67,
            "cutoffs_distinct": bool(
                q33 < q67
            ),
            "n_valid_observations": int(
                df[value_col].notna().sum()
            ),
            "n_unique_values": int(
                df[value_col].nunique(
                    dropna=True
                )
            )
        }
    )

group_cutoffs = pd.DataFrame(
    group_cutoff_rows
)


# =========================
# Person-specific tertiles
# =========================
df, pa_person_cutoffs = (
    add_person_tertiles(
        df,
        ID_COL,
        PA_COL,
        "PA"
    )
)

df, na_person_cutoffs = (
    add_person_tertiles(
        df,
        ID_COL,
        NA_COL,
        "NA"
    )
)


# =========================
# Joint PA x NA states
# =========================
df["Affect_state_fixed"] = (
    combine_pa_na(
        df["PA_state_fixed"],
        df["NA_state_fixed"]
    )
)

df["Affect_state_group_tertile"] = (
    combine_pa_na(
        df["PA_state_group_tertile"],
        df["NA_state_group_tertile"]
    )
)

df["Affect_state_person_tertile"] = (
    combine_pa_na(
        df["PA_state_person_tertile"],
        df["NA_state_person_tertile"]
    )
)

df[
    "Affect_state_person_tertile_strict"
] = combine_pa_na(
    df[
        "PA_state_person_tertile_strict"
    ],
    df[
        "NA_state_person_tertile_strict"
    ]
)


# =========================
# Distribution and agreement summaries
# =========================
method_columns = {
    "PA": {
        "fixed": (
            "PA_state_fixed"
        ),
        "group": (
            "PA_state_group_tertile"
        ),
        "person": (
            "PA_state_person_tertile"
        )
    },

    "NA": {
        "fixed": (
            "NA_state_fixed"
        ),
        "group": (
            "NA_state_group_tertile"
        ),
        "person": (
            "NA_state_person_tertile"
        )
    }
}

method_labels = {
    "fixed": "Fixed threshold",
    "group": "Group tertiles",
    "person": "Person-specific tertiles"
}

distribution_rows = []
agreement_rows = []
confusion_results = {}

for emotion, methods in method_columns.items():

    for method_key, col in methods.items():

        n_valid = int(
            df[col].notna().sum()
        )

        n_missing = int(
            df[col].isna().sum()
        )

        for state in LEVELS:

            n_state = int(
                (
                    df[col].astype("object")
                    == state
                ).sum()
            )

            if n_valid > 0:
                pct = (
                    n_state
                    / n_valid
                    * 100
                )
            else:
                pct = np.nan

            distribution_rows.append(
                {
                    "emotion": emotion,
                    "method": method_key,
                    "method_label": (
                        method_labels[method_key]
                    ),
                    "state": state,
                    "n": n_state,
                    "percent_among_valid": pct,
                    "n_valid": n_valid,
                    "n_missing": n_missing
                }
            )

    for (
        method_a,
        col_a
    ), (
        method_b,
        col_b
    ) in combinations(
        methods.items(),
        2
    ):

        tmp = df[
            [
                col_a,
                col_b
            ]
        ].dropna()

        n_compared = len(tmp)

        if n_compared > 0:

            exact_agreement = (
                (
                    tmp[col_a].astype("object")
                    == tmp[col_b].astype("object")
                )
                .mean()
                * 100
            )

            extreme_disagreement = (
                (
                    (
                        tmp[col_a].astype("object")
                        == "Low"
                    )
                    & (
                        tmp[col_b].astype("object")
                        == "High"
                    )
                )
                |
                (
                    (
                        tmp[col_a].astype("object")
                        == "High"
                    )
                    & (
                        tmp[col_b].astype("object")
                        == "Low"
                    )
                )
            ).mean() * 100

        else:
            exact_agreement = np.nan
            extreme_disagreement = np.nan

        kappa = linear_weighted_kappa(
            df[col_a],
            df[col_b]
        )

        agreement_rows.append(
            {
                "emotion": emotion,
                "method_A": method_a,
                "method_B": method_b,
                "comparison": (
                    f"{method_labels[method_a]}"
                    " vs "
                    f"{method_labels[method_b]}"
                ),
                "n_compared": n_compared,
                "exact_agreement_percent": (
                    exact_agreement
                ),
                "extreme_disagreement_percent": (
                    extreme_disagreement
                ),
                "linear_weighted_kappa": (
                    kappa
                )
            }
        )

        counts, row_pct = confusion_tables(
            df[col_a],
            df[col_b]
        )

        result_key = (
            f"{emotion}_"
            f"{method_a}_"
            f"{method_b}"
        )

        confusion_results[result_key] = {
            "counts": counts,
            "row_percent": row_pct,
            "emotion": emotion,
            "method_a": method_a,
            "method_b": method_b
        }


distribution_summary = pd.DataFrame(
    distribution_rows
)

agreement_summary = pd.DataFrame(
    agreement_rows
)


# =========================
# Person-specific tertile availability
# =========================
person_validity_summary = pd.DataFrame(
    [
        {
            "emotion": "PA",

            "n_participants": len(
                pa_person_cutoffs
            ),

            "n_valid_three_levels": int(
                pa_person_cutoffs[
                    "PA_person_3level_valid"
                ].sum()
            ),

            "percent_valid_three_levels": (
                pa_person_cutoffs[
                    "PA_person_3level_valid"
                ].mean()
                * 100
            ),

            "n_q33_equal_q67": int(
                (
                    pa_person_cutoffs[
                        "PA_person_q33"
                    ]
                    ==
                    pa_person_cutoffs[
                        "PA_person_q67"
                    ]
                ).sum()
            ),

            "n_fewer_than_3_unique_values": int(
                (
                    pa_person_cutoffs[
                        "PA_person_n_unique"
                    ]
                    < 3
                ).sum()
            )
        },

        {
            "emotion": "NA",

            "n_participants": len(
                na_person_cutoffs
            ),

            "n_valid_three_levels": int(
                na_person_cutoffs[
                    "NA_person_3level_valid"
                ].sum()
            ),

            "percent_valid_three_levels": (
                na_person_cutoffs[
                    "NA_person_3level_valid"
                ].mean()
                * 100
            ),

            "n_q33_equal_q67": int(
                (
                    na_person_cutoffs[
                        "NA_person_q33"
                    ]
                    ==
                    na_person_cutoffs[
                        "NA_person_q67"
                    ]
                ).sum()
            ),

            "n_fewer_than_3_unique_values": int(
                (
                    na_person_cutoffs[
                        "NA_person_n_unique"
                    ]
                    < 3
                ).sum()
            )
        }
    ]
)


# =========================
# Plotting functions
# =========================
def plot_state_distribution(emotion):
    """Plot state distributions across classification methods."""

    plot_df = (
        distribution_summary[
            distribution_summary["emotion"]
            == emotion
        ]
        .pivot(
            index="method_label",
            columns="state",
            values="percent_among_valid"
        )
        .reindex(columns=LEVELS)
        .reindex(
            [
                method_labels["fixed"],
                method_labels["group"],
                method_labels["person"]
            ]
        )
    )

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5)
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "Percentage of valid observations"
    )
    ax.set_ylim(0, 100)

    ax.set_title(
        f"{emotion} state distribution "
        "by classification method"
    )

    ax.legend(
        title="State",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / f"{emotion}_state_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_kappa(emotion):
    """Plot linear weighted kappa across method comparisons."""

    plot_df = agreement_summary[
        agreement_summary["emotion"]
        == emotion
    ].copy()

    plt.figure(
        figsize=(8, 5)
    )

    ax = plt.gca()

    ax.bar(
        plot_df["comparison"],
        plot_df["linear_weighted_kappa"]
    )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.set_ylim(
        -1,
        1
    )

    ax.set_ylabel(
        "Linear weighted kappa"
    )

    ax.set_title(
        f"{emotion}: agreement between "
        "classification methods"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / f"{emotion}_weighted_kappa.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


for emotion in ["PA", "NA"]:

    plot_state_distribution(emotion)

    plot_kappa(emotion)


# =========================
# Save Excel output
# =========================
notes = pd.DataFrame(
    {
        "column_or_output": [
            "PA/NA_state_fixed",

            "PA/NA_state_group_tertile",

            "PA/NA_state_person_tertile",

            (
                "PA/NA_state_"
                "person_tertile_strict"
            ),

            "Affect_state_*",

            "linear_weighted_kappa",

            "confusion matrices"
        ],

        "description": [
            (
                "Fixed thresholds: values <= 33 are classified as Low, "
                "values > 33 and <= 66 as Medium, and values > 66 as High."
            ),

            (
                "Classification based on q33 and q67 from all valid EMA observations."
            ),

            (
                "Classification based on each participant's own q33 and q67; "
                "some participants may have only one or two observed state levels."
            ),

            (
                "Strict person-specific tertile classification; values are retained only "
                "for participants with all three observed state levels."
            ),

            (
                "Joint states formed by combining PA and NA classifications."
            ),

            (
                "Agreement index for ordered classifications."
            ),

            (
                "Count and row-percentage matrices comparing two classification methods."
            )
        ]
    }
)


with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="data_all_states",
        index=False
    )

    group_cutoffs.to_excel(
        writer,
        sheet_name="group_cutoffs",
        index=False
    )

    pa_person_cutoffs.to_excel(
        writer,
        sheet_name="PA_person_cutoffs",
        index=False
    )

    na_person_cutoffs.to_excel(
        writer,
        sheet_name="NA_person_cutoffs",
        index=False
    )

    person_validity_summary.to_excel(
        writer,
        sheet_name="person_validity",
        index=False
    )

    distribution_summary.to_excel(
        writer,
        sheet_name="state_distribution",
        index=False
    )

    agreement_summary.to_excel(
        writer,
        sheet_name="agreement_summary",
        index=False
    )

    notes.to_excel(
        writer,
        sheet_name="notes",
        index=False
    )

    for key, result in confusion_results.items():

        sheet_name = key[:31]

        counts = result["counts"].copy()
        counts.index.name = "Method_A_count"

        row_pct = (
            result["row_percent"].copy()
        )
        row_pct.index.name = (
            "Method_A_row_percent"
        )

        counts.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=0,
            startcol=0
        )

        row_pct.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=0,
            startcol=6
        )


# =========================
# Print results
# =========================
print("\nDone.")

print(
    f"Classified data and summary results:\n{OUTPUT_FILE}"
)

print(
    f"\nFigure directory:\n{FIG_DIR}"
)

print("\nGroup-level tertile cutoffs:")

print(
    group_cutoffs.to_string(
        index=False
    )
)

print("\nPerson-specific tertile availability:")

print(
    person_validity_summary.to_string(
        index=False
    )
)

print("\nAgreement across classification methods:")

print(
    agreement_summary.to_string(
        index=False
    )
)
