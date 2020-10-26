# coding: utf-8
import pandas as pd
import numpy as np
import os
import seaborn as sns
import re
import matplotlib.pyplot as plt
import matplotlib
import table_utils

sns.set(font_scale=1.3)
styles = [
    "seaborn-paper",
    "seaborn",
    "seaborn-white",
]  # ['seaborn-paper','seaborn', 'seaborn-white']
plt.style.use(styles)
sns.set_context("paper", font_scale=1.6)
matplotlib.rc("font", family="Times New Roman")
CSV_PATH = os.path.join(".", "csv")
FIGURES_PATH = os.path.join(".", "plots")
if not os.path.exists(FIGURES_PATH):
    os.makedirs(FIGURES_PATH)


# full_fcn = pd.read_csv(
#     os.path.join(CSV_PATH, "final_iou_full_component_tagged_fcn.csv")
# )
# full_sw = pd.read_csv(os.path.join(CSV_PATH, "final_iou_full_component_tagged_sw.csv"))


precision_recall = pd.read_csv(
    os.path.join(CSV_PATH, "final_iou_full_precision_recall.csv")
)
precision_recall = table_utils.get_arch_col(precision_recall)
precision_recall = precision_recall.fillna(value=0)
precision_recall = precision_recall[precision_recall.iou_threshold == 0.5]
sw = precision_recall.loc[precision_recall.architecture == "SW"]
fcn = precision_recall.loc[precision_recall.architecture == "FCN"]

sns.scatterplot(
    data=sw,
    x="recall",
    y="precision",
    color="w",
    edgecolor="black",
    label="SW",
    linewidth=1,
    s=40,
)
sns.scatterplot(
    data=fcn,
    x="recall",
    y="precision",
    color="black",
    edgecolor="black",
    label="FCN-MN",
    s=40,
)
plt.legend(frameon=True)
plt.xlabel("Detection Recall")
plt.ylabel("Detection Precision")
plt.ylim((0, 1))
plt.xlim((0, 1))
plt.title("Detection Precision and Recall for IoU $\geq$ 0.5")
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_PATH, "Figure3-a.png"), dpi=250)

plt.clf()
# Mismo plot para IoU >= 0.1

precision_recall = pd.read_csv(
    os.path.join(CSV_PATH, "final_iou_full_precision_recall.csv")
)
precision_recall = table_utils.get_arch_col(precision_recall)
precision_recall = precision_recall.fillna(value=0)
precision_recall = precision_recall[precision_recall.iou_threshold == 0.1]
sw = precision_recall.loc[precision_recall.architecture == "SW"]
fcn = precision_recall.loc[precision_recall.architecture == "FCN"]

sns.scatterplot(
    data=sw,
    x="recall",
    y="precision",
    color="w",
    edgecolor="black",
    label="SW",
    linewidth=1,
    s=40,
)
sns.scatterplot(
    data=fcn,
    x="recall",
    y="precision",
    color="black",
    edgecolor="black",
    label="FCN-MN",
    s=40,
)
plt.legend(frameon=True)
plt.xlabel("Detection Recall")
plt.ylabel("Detection Precision")
plt.ylim((0, 1))
plt.xlim((0, 1))
plt.title("Detection Precision and Recall for IoU $\geq$ 0.1")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "Figure3-b.png"), dpi=250)

plt.clf()


# Histograma de splits para comparar FCN vs SW. Igual q el mandastes anoche


def get_split_distribution_barplot(data, models=None):
    bins = np.linspace(0, 225, 10, endpoint=True).astype(int)
    fragments = []
    for model in data.architecture.unique():
        temp = data.loc[data.architecture == model].splits
        frag = pd.cut(temp, bins=bins).value_counts().reset_index()
        if model == "FCN":
            frag["model"] = "FCN-MN"
        else:
            frag["model"] = "SW"
        frag["splits"] = frag["splits"] / frag["splits"].sum()
        fragments.append(frag)
    full = pd.concat(fragments, axis=0)
    # Custom colors
    colors = ["#000000", "#FFFFFF"]
    custom_palette = sns.set_palette(sns.color_palette(colors))

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=full,
        x="index",
        y="splits",
        hue="model",
        palette=custom_palette,
        edgecolor="black",
        linewidth=1.5,
    )
    plt.xlabel("Number of Split Components")
    plt.ylabel("Proportion")
    plt.ylim((0, 1))
    plt.xticks(rotation=30)
    plt.legend(title="Models", frameon=True)
    plt.title("Number of Split Components for FCN-MN and Sliding Windows")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "Figure4.png"), dpi=250)


detection_report = pd.read_csv(os.path.join(CSV_PATH, "rev_05iou_detection.csv"))
detection_report = table_utils.get_arch_col(detection_report)
detection_report.groupby(["model_name", "threshold_pw"]).sum()["splits"].max()
get_split_distribution_barplot(detection_report)

plt.clf()

# Histograma de Distancia Normalizada

bins = np.linspace(0, 60, 61).astype(int)

full_area_df = table_utils.get_arch_col(
    pd.read_csv(os.path.join(CSV_PATH, "rev_05iou_segmentation_na_raw.csv"))
)

fcn_mean_area = full_area_df[full_area_df.architecture == "FCN"][
    "relative_area_to_gt_mean"
]
fcn_mean_area = pd.cut(fcn_mean_area, bins=bins).value_counts().reset_index()
fcn_mean_area["relative_area_to_gt_mean"] = (
    fcn_mean_area["relative_area_to_gt_mean"]
    / fcn_mean_area["relative_area_to_gt_mean"].sum()
)
fcn_mean_area["metamodel"] = "FCN-MN"

sw_mean_area = full_area_df[full_area_df.architecture == "SW"][
    "relative_area_to_gt_mean"
]
sw_mean_area = pd.cut(sw_mean_area, bins=bins).value_counts().reset_index()
sw_mean_area["relative_area_to_gt_mean"] = (
    sw_mean_area["relative_area_to_gt_mean"]
    / sw_mean_area["relative_area_to_gt_mean"].sum()
)
sw_mean_area["metamodel"] = "SW"

full_area_df = pd.concat([fcn_mean_area, sw_mean_area], axis=0)

colors = ["#000000", "#FFFFFF"]
custom_palette = sns.set_palette(sns.color_palette(colors))
sns.barplot(
    data=full_area_df,
    x="index",
    y="relative_area_to_gt_mean",
    hue="metamodel",
    palette=custom_palette,
    edgecolor="black",
    linewidth=1,
)
plt.legend(frameon=True)
bins = np.linspace(0, 60, 13).astype(int)
plt.xticks(bins, bins)
plt.xlabel("Mean Normalized Area ")
plt.ylabel("Normalized Count")
plt.title("Mean Normalized Area of False Alarm Components")
plt.savefig(os.path.join(FIGURES_PATH, "Figure6.png"), dpi=250)
plt.clf()


# Histograma de DISTANCIA relativa promedio de los false alarms FCN vs SW


full_distance_df = table_utils.get_arch_col(
    pd.read_csv(os.path.join(CSV_PATH, "final_rev_05iou_localization_raw.csv"))
)
bins = np.linspace(0, 10, 11).astype(int)
fcn_mean_distance = full_distance_df[full_distance_df.architecture == "FCN"][
    "norm_distance_mean"
]
fcn_mean_distance = pd.cut(fcn_mean_distance, bins=bins).value_counts().reset_index()
fcn_mean_distance["norm_distance_mean"] = (
    fcn_mean_distance["norm_distance_mean"]
    / fcn_mean_distance["norm_distance_mean"].sum()
)
fcn_mean_distance["metamodel"] = "FCN-MN"

sw_mean_distance = full_distance_df[full_distance_df.architecture == "SW"][
    "norm_distance_mean"
]
sw_mean_distance = pd.cut(sw_mean_distance, bins=bins).value_counts().reset_index()
sw_mean_distance["norm_distance_mean"] = (
    sw_mean_distance["norm_distance_mean"]
    / sw_mean_distance["norm_distance_mean"].sum()
)
sw_mean_distance["metamodel"] = "SW"

full_distance_df = pd.concat([fcn_mean_distance, sw_mean_distance], axis=0)

bins = np.arange(0, 10.5, 1)
colors = ["#000000", "#FFFFFF"]
custom_palette = sns.set_palette(sns.color_palette(colors))
sns.barplot(
    data=full_distance_df,
    x="index",
    y="norm_distance_mean",
    hue="metamodel",
    palette=custom_palette,
    edgecolor="black",
    linewidth=1,
)
plt.legend(frameon=True)
plt.xticks(bins, rotation=30)
plt.xlabel("Mean Normalized Distance")
plt.ylabel("Normalized Count")
plt.ylim((0, 1))
plt.tight_layout()
plt.title("Mean Normalized Distance of False Alarm Components")
plt.savefig(os.path.join(FIGURES_PATH, "Figure7.png"), dpi=250)

plt.clf()

# Scatterplots de Segmentacion con Errorbars
# TRUE POSITIVES

precision_recall = table_utils.get_arch_col(
    pd.read_csv(os.path.join(CSV_PATH, "rev_05iou_segmentation_cd_raw.csv"))
)
fcn = precision_recall[precision_recall.architecture == "FCN"]
sw = precision_recall[precision_recall.architecture == "SW"]

sns.scatterplot(
    x=fcn["component_recall_joint_mean"],
    y=fcn["component_precision_joint_mean"],
    label="FCN-MN",
    color="black",
    edgecolor="black",
    linewidth=1,
    s=40,
)
sns.scatterplot(
    x=sw["component_recall_joint_mean"],
    y=sw["component_precision_joint_mean"],
    color="w",
    edgecolor="black",
    label="SW",
    linewidth=1,
    s=40,
)


plt.errorbar(
    fcn["component_recall_joint_mean"],
    fcn["component_precision_joint_mean"],
    xerr=fcn["component_recall_joint_std"],
    yerr=fcn["component_precision_joint_std"],
    fmt="ko",
    elinewidth=0.5,
    capthick=0.1,
    ecolor="gray",
    color="black",
)

plt.errorbar(
    sw["component_recall_joint_mean"],
    sw["component_precision_joint_mean"],
    xerr=sw["component_recall_joint_std"],
    yerr=sw["component_precision_joint_std"],
    fmt="wo",
    ecolor="gray",
    capthick=0.1,
    elinewidth=0.5,
    linewidth=1,
    color="white",
)

plt.title("Mean Segmentation Precision vs Recall for True Positive Components")
plt.xlabel("Mean Segmentation Recall")
plt.ylabel("Mean Segmentation Precision")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "Figure5-a.png"), dpi=250)


plt.clf()
# SPLITS
precision_recall = table_utils.get_arch_col(
    pd.read_csv(os.path.join(CSV_PATH, "rev_05iou_segmentation_split_raw.csv"))
)

fcn = precision_recall[precision_recall.architecture == "FCN"]
sw = precision_recall[precision_recall.architecture == "SW"]

sns.scatterplot(
    x=fcn["component_recall_joint_mean"],
    y=fcn["component_precision_joint_mean"],
    label="FCN-MN",
    color="black",
    edgecolor="black",
    linewidth=1,
    s=40,
)
sns.scatterplot(
    x=sw["component_recall_joint_mean"],
    y=sw["component_precision_joint_mean"],
    color="w",
    edgecolor="black",
    label="SW",
    linewidth=1,
    s=40,
)

plt.errorbar(
    fcn["component_recall_joint_mean"],
    fcn["component_precision_joint_mean"],
    xerr=fcn["component_recall_joint_std"],
    yerr=fcn["component_precision_joint_std"],
    fmt="ko",
    elinewidth=0.5,
    capthick=0.1,
    ecolor="gray",
    color="black",
)

plt.errorbar(
    sw["component_recall_joint_mean"],
    sw["component_precision_joint_mean"],
    xerr=sw["component_recall_joint_std"],
    yerr=sw["component_precision_joint_std"],
    fmt="wo",
    ecolor="gray",
    capthick=0.1,
    elinewidth=0.5,
    linewidth=1,
    color="white",
)

plt.title("Mean Segmentation Precision vs Recall for Split Components")
plt.xlabel("Mean Segmentation Recall")
plt.ylabel("Mean Segmentation Precision")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, "Figure5-b.png"), dpi=250)
plt.clf()
