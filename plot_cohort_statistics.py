import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager


# ---------- Global style ----------
def select_serif_font():
    candidates = [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "DejaVu Serif",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            return font_name
    return "DejaVu Serif"


IEEE_SERIF_FONT = select_serif_font()

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [IEEE_SERIF_FONT, "Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.titlesize"] = 9.5
plt.rcParams["axes.labelsize"] = 8.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.labelsize"] = 8.2
plt.rcParams["ytick.labelsize"] = 8.2


REQUIRED_COLUMNS = [
    "case_id",
    "ihc_file",
    "he_file",
    "pathological_response",
    "label",
    "age",
    "pathology.type",
]


PATHOLOGY_LEGEND_LABELS = {
    "1": "LUAD",
    "2": "LUSC",
    "3": "Others",
}


def strip_slide_ext(value):
    value = os.path.basename(str(value))
    return re.sub(r"\.(svs|tif|tiff|kfb|jpeg|jpg|png)$", "", value, flags=re.IGNORECASE)


def normalize_file(value):
    if pd.isna(value):
        return ""
    return strip_slide_ext(value).strip().lower()


def normalize_response_from_label(label):
    if pd.isna(label):
        return "Unknown"
    try:
        label = int(float(label))
    except Exception:
        return "Unknown"
    return "pCR" if label == 1 else "non-pCR"


def clean_pathology_type(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Unknown"
    value_str = str(value).strip()
    value_lower = value_str.lower()
    if value_lower in ["nan", "none", "unknown", "na", "n/a", "-"]:
        return "Unknown"
    try:
        numeric_value = float(value_str)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except Exception:
        pass
    return value_str


def pathology_legend_label(category):
    return PATHOLOGY_LEGEND_LABELS.get(str(category).strip(), str(category))


def load_one_dataset(path, dataset_key):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}. Current columns: {list(df.columns)}")

    keep_cols = [
        "case_id",
        "ihc_file",
        "he_file",
        "pathological_response",
        "label",
        "gender",
        "age",
        "PDL1.TPS",
        "smoking.status",
        "pathology.type",
        "T.stage",
        "N.stage",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols].copy()

    df["dataset_orig"] = dataset_key
    df["case_id"] = df["case_id"].astype(str)
    df["he_file"] = df["he_file"].astype(str)
    df["ihc_file"] = df["ihc_file"].astype(str)
    df["he_stem"] = df["he_file"].apply(normalize_file)
    df["ihc_stem"] = df["ihc_file"].apply(normalize_file)
    df["he_ihc_pair_key"] = df["case_id"].str.lower() + "::" + df["he_stem"] + "__" + df["ihc_stem"]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["response"] = df["label"].apply(normalize_response_from_label)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["pathology_type_clean"] = df["pathology.type"].apply(clean_pathology_type)
    return df


def prepare_data(args):
    df_gd = load_one_dataset(args.cohort1_csv, "Cohort1")
    df_zd = load_one_dataset(args.cohort2_csv, "Cohort2")
    pair_df = pd.concat([df_gd, df_zd], ignore_index=True)

    # Pair-level table: one row per HE-IHC pair.
    pair_df = pair_df.drop_duplicates(subset=["dataset_orig", "he_ihc_pair_key"], keep="first").reset_index(drop=True)
    pair_df["case_key_global"] = pair_df["dataset_orig"].astype(str) + "::" + pair_df["case_id"].astype(str)
    pair_df["pair_key_global"] = pair_df["dataset_orig"].astype(str) + "::" + pair_df["he_ihc_pair_key"].astype(str)

    # Case-level table: one row per unique case, used for clinical feature distribution.
    case_df = pair_df.drop_duplicates(subset=["case_key_global"], keep="first").reset_index(drop=True)
    return pair_df, case_df


def summarize_counts(pair_df, case_df, out_dir):
    rows = []
    for dataset in ["Cohort1", "Cohort2", "Overall"]:
        if dataset == "Overall":
            pair_sub = pair_df
            case_sub = case_df
            display_name = "Overall"
        else:
            pair_sub = pair_df[pair_df["dataset_orig"] == dataset]
            case_sub = case_df[case_df["dataset_orig"] == dataset]
            display_name = dataset

        label_counts_case = case_sub["response"].value_counts().to_dict()
        label_counts_pair = pair_sub["response"].value_counts().to_dict()
        rows.append(
            {
                "Dataset": display_name,
                "Unique cases": int(case_sub["case_key_global"].nunique()),
                "Unique HE-IHC pairs": int(pair_sub["pair_key_global"].nunique()),
                "Case-level pCR": int(label_counts_case.get("pCR", 0)),
                "Case-level non-pCR": int(label_counts_case.get("non-pCR", 0)),
                "Pair-level pCR": int(label_counts_pair.get("pCR", 0)),
                "Pair-level non-pCR": int(label_counts_pair.get("non-pCR", 0)),
            }
        )

    summary_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "clinical_data_summary_counts.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"[DONE] Saved summary: {summary_path}")
    return summary_df


def stacked_percent_table(df, dataset_col, category_col, order, category_order=None):
    pivot = df.groupby([dataset_col, category_col]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=order, fill_value=0)
    if category_order is not None:
        pivot = pivot.reindex(columns=category_order, fill_value=0)
    totals = pivot.sum(axis=1).replace(0, np.nan)
    return pivot.div(totals, axis=0) * 100


def apply_ieee_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8, pad=2)
    ax.grid(True, axis="y", color="#D9D9D9", alpha=0.75, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)


def contrast_text_color(facecolor):
    r, g, b, _ = facecolor
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "black"


def style_stacked_bars(ax):
    for container in ax.containers:
        for patch in container:
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)


def annotate_stacked_bars(ax, labels=None, min_height=4.0, label_min_height=9.0):
    for container_idx, container in enumerate(ax.containers):
        label = labels[container_idx] if labels is not None and container_idx < len(labels) else None
        for patch in container:
            height = patch.get_height()
            if height >= min_height:
                text = f"{height:.1f}%"
                if label and height >= label_min_height:
                    text = f"{label}\n{height:.1f}%"
                ax.text(
                    patch.get_x() + patch.get_width() / 2.0,
                    patch.get_y() + height / 2.0,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=contrast_text_color(patch.get_facecolor()),
                    weight="normal",
                    linespacing=0.95,
                )


def plot_and_save(case_df, name_mapping, suffix, out_dir):
    df_plot = case_df.copy()
    df_plot["dataset"] = df_plot["dataset_orig"].map(name_mapping)
    order = [name_mapping["Cohort1"], name_mapping["Cohort2"]]

    pathology_counts = df_plot["pathology_type_clean"].value_counts()
    pathology_order = [cat for cat in ["1", "2", "3"] if cat in pathology_counts.index]
    pathology_order.extend(
        [
            cat
            for cat in pathology_counts.index.tolist()
            if cat not in pathology_order and cat != "Unknown"
        ]
    )
    if "Unknown" in pathology_counts.index:
        pathology_order.append("Unknown")

    response_order = ["non-pCR", "pCR"]
    pathology_pct = stacked_percent_table(
        df_plot,
        dataset_col="dataset",
        category_col="pathology_type_clean",
        order=order,
        category_order=pathology_order,
    )
    response_pct = stacked_percent_table(
        df_plot,
        dataset_col="dataset",
        category_col="response",
        order=order,
        category_order=response_order,
    )

    dataset_palette = ["#3F7FBF", "#C98549"]
    response_colors = {"non-pCR": "#C95F4A", "pCR": "#4F9A5B"}
    pathology_palette = ["#7B6FB6", "#55AFA8", "#C49A49", "#B97993", "#91A65F", "#96735D", "#6D9BC3"]
    pathology_colors = {cat: pathology_palette[i % len(pathology_palette)] for i, cat in enumerate(pathology_order)}
    if "Unknown" in pathology_colors:
        pathology_colors["Unknown"] = "#A8A8A8"

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.9))
    fig.subplots_adjust(wspace=0.36, right=0.98, left=0.08, bottom=0.17)

    # 1. Age distribution.
    ax = axes[0]
    sns.boxplot(
        data=df_plot,
        x="dataset",
        y="age",
        hue="dataset",
        palette=dataset_palette,
        width=0.5,
        ax=ax,
        legend=False,
        order=order,
        linewidth=0.8,
        fliersize=2,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
    )
    sns.stripplot(
        data=df_plot,
        x="dataset",
        y="age",
        color="black",
        alpha=0.3,
        size=2,
        ax=ax,
        order=order,
    )
    ax.set_title("(a) Age distribution", fontweight="bold", pad=5)
    ax.set_ylabel("Age (years)")
    ax.set_xlabel("")
    apply_ieee_axis(ax)

    # 2. Pathology type distribution.
    ax = axes[1]
    pathology_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[pathology_colors[c] for c in pathology_pct.columns],
        legend=False,
        width=0.62,
    )
    style_stacked_bars(ax)
    ax.set_title("(b) Pathology type", fontweight="bold", pad=5)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Cohort")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)
    apply_ieee_axis(ax)
    pathology_labels = [pathology_legend_label(c) for c in pathology_pct.columns]
    annotate_stacked_bars(ax, labels=pathology_labels, min_height=5.0, label_min_height=5.0)

    # 3. Positive/negative sample distribution.
    ax = axes[2]
    response_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[response_colors[c] for c in response_pct.columns],
        legend=False,
        width=0.62,
    )
    style_stacked_bars(ax)
    ax.set_title("(c) Response", fontweight="bold", pad=5)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)
    apply_ieee_axis(ax)
    response_labels = list(response_pct.columns)
    annotate_stacked_bars(ax, labels=response_labels, min_height=4.0, label_min_height=4.0)

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"clinical_distribution_{suffix}.png")
    pdf_path = os.path.join(out_dir, f"clinical_distribution_{suffix}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved: {png_path}")
    print(f"[DONE] Saved: {pdf_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize clinical data distributions for HE-IHC cohorts.")
    parser.add_argument("--cohort1_csv", type=str, required=True)
    parser.add_argument("--cohort2_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./clinical_distribution_figures")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Using serif font: {IEEE_SERIF_FONT}")
    pair_df, case_df = prepare_data(args)
    summarize_counts(pair_df, case_df, args.out_dir)

    name_mapping_real = {"Cohort1": "Cohort 1", "Cohort2": "Cohort 2"}
    name_mapping_anon = {"Cohort1": "Cohort 1", "Cohort2": "Cohort 2"}

    plot_and_save(case_df, name_mapping_real, "real", args.out_dir)
    plot_and_save(case_df, name_mapping_anon, "anon", args.out_dir)


if __name__ == "__main__":
    main()
