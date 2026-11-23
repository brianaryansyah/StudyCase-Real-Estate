"""
Membuat grafik validasi untuk laporan studi kasus valuasi real estat.

Jalankan dari akar repositori:
    python scripts/generate_charts.py

Keluaran berupa berkas PNG di direktori images/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Real estate valuation data set.xlsx"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = pd.read_excel(DATA_PATH)
    target_col = "Y house price of unit area"
    feature_cols = [c for c in df.columns if c not in ["No", target_col]]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin_reg = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_lin = lin_reg.predict(X_test_scaled)
    metrics_lin = evaluate_model("Regresi Linier", y_test, y_pred_lin)

    k_values = [2, 3, 5, 7, 9]
    results_knn = []
    models_knn = {}
    for k in k_values:
        model = KNeighborsRegressor(n_neighbors=k).fit(X_train_scaled, y_train)
        models_knn[k] = model
        results_knn.append(
            evaluate_model(f"k-NN (k={k})", y_test, model.predict(X_test_scaled))
        )
    tabel_knn = pd.DataFrame(results_knn)
    results_all = pd.concat([pd.DataFrame([metrics_lin]), tabel_knn], ignore_index=True)

    best_idx = results_all["RMSE"].idxmin()
    best_name = results_all.loc[best_idx, "Model"]
    best_k = int(best_name.split("=")[1].rstrip(")"))
    y_pred_best = models_knn[best_k].predict(X_test_scaled)
    best_metrics = results_all.loc[best_idx]

    # 1. Distribusi target dan korelasi fitur
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Validasi Soal 1.a: Distribusi Target dan Korelasi Fitur",
        fontsize=16,
        fontweight="bold",
    )
    ax = axes[0]
    sns.histplot(y, bins=25, kde=True, color="#2E86AB", ax=ax)
    ax.axvline(y.mean(), color="#C73E1D", linestyle="--", linewidth=1.8,
               label=f"Rata-rata = {y.mean():.2f}")
    ax.axvline(y.median(), color="#3A7D44", linestyle=":", linewidth=1.8,
               label=f"Median = {y.median():.2f}")
    ax.set_title("Distribusi Harga per Satuan Area")
    ax.set_xlabel("Harga per satuan area")
    ax.set_ylabel("Frekuensi")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    corr = df[feature_cols + [target_col]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriks Korelasi Fitur dan Target")
    ax.tick_params(axis="x", rotation=42, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(IMAGE_DIR / "01_distribusi_dan_korelasi.png")
    plt.close(fig)

    # 2. Koefisien regresi dan kecocokan prediksi
    coef_df = pd.DataFrame(
        {"Fitur": feature_cols, "Koefisien": lin_reg.coef_}
    ).sort_values("Koefisien")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle("Validasi Soal 1.b: Persamaan Regresi Linier",
                 fontsize=16, fontweight="bold")
    ax = axes[0]
    warna = ["#C73E1D" if v < 0 else "#3A7D44" for v in coef_df["Koefisien"]]
    ax.barh(coef_df["Fitur"], coef_df["Koefisien"], color=warna,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Koefisien Regresi Terstandardisasi")
    ax.set_xlabel("Nilai koefisien")
    ax.set_ylabel("Fitur")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.scatter(y_test, y_pred_lin, alpha=0.75, s=45, color="#2E86AB",
               edgecolor="black", linewidth=0.4)
    batas = [min(float(y_test.min()), float(y_pred_lin.min())),
             max(float(y_test.max()), float(y_pred_lin.max()))]
    ax.plot(batas, batas, "r--", linewidth=1.8, label="Garis identitas (y = x)")
    ax.set_title("Nilai Aktual dan Nilai Prediksi")
    ax.set_xlabel("Harga aktual per satuan area")
    ax.set_ylabel("Harga prediksi per satuan area")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMAGE_DIR / "02_koefisien_regresi.png")
    plt.close(fig)

    # 3. Metrik dan residual regresi linier
    residual_lin = y_test.values - y_pred_lin
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Validasi Soal 1.c: Kinerja dan Residual Regresi Linier",
                 fontsize=16, fontweight="bold")
    panel = [("RMSE", metrics_lin["RMSE"], "#2E86AB"),
             ("MAE", metrics_lin["MAE"], "#F18F01"),
             ("MAPE (%)", metrics_lin["MAPE (%)"], "#6A4C93"),
             ("R2", metrics_lin["R2"], "#3A7D44")]
    for ax, (label, nilai, warna_bar) in zip(axes.ravel()[:4], panel):
        bar = ax.bar([label], [nilai], color=warna_bar, edgecolor="black", linewidth=0.4)
        ax.set_title(f"Metrik {label}")
        ax.set_ylabel(label)
        ax.bar_label(bar, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.0 if label == "R2" else max(nilai * 1.25, 1.0))
        ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(y_pred_lin, residual_lin, alpha=0.75, s=45, color="#F18F01",
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="#C73E1D", linestyle="--", linewidth=1.8)
    ax.set_title("Residual terhadap Nilai Prediksi")
    ax.set_xlabel("Harga prediksi per satuan area")
    ax.set_ylabel("Residual (aktual - prediksi)")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    sns.histplot(residual_lin, bins=20, kde=True, color="#6A4C93", ax=ax)
    ax.axvline(0, color="#C73E1D", linestyle="--", linewidth=1.8)
    ax.set_title("Distribusi Residual")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(IMAGE_DIR / "03_metrik_regresi.png")
    plt.close(fig)

    # 4. Pengaruh nilai k
    k_axis = k_values
    metrik_knn = [("RMSE", "#C73E1D", "min"), ("MAE", "#F18F01", "min"),
                  ("MAPE (%)", "#6A4C93", "min"), ("R2", "#3A7D44", "max")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Validasi Soal 2: Pengaruh Nilai k terhadap Kinerja k-NN",
                 fontsize=16, fontweight="bold")
    for ax, (kolom, warna_line, arah) in zip(axes.ravel(), metrik_knn):
        nilai = tabel_knn[kolom].to_numpy()
        ax.plot(k_axis, nilai, marker="o", markersize=8, linewidth=2.2,
                color=warna_line, label="k-NN")
        ax.axhline(metrics_lin[kolom], color="#2E86AB", linestyle="--",
                   linewidth=1.8, label="Regresi Linier")
        idx_best = int(np.argmin(nilai) if arah == "min" else np.argmax(nilai))
        ax.scatter(k_axis[idx_best], nilai[idx_best], marker="*", s=260,
                   color="#FFD166", edgecolor="black", linewidth=0.8, zorder=5,
                   label=f"k terbaik = {k_axis[idx_best]}")
        for x_val, y_val in zip(k_axis, nilai):
            ax.annotate(f"{y_val:.4f}", (x_val, y_val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)
        ax.set_title(f"{kolom} terhadap Nilai k")
        ax.set_xlabel("Nilai k")
        ax.set_ylabel(kolom)
        ax.set_xticks(k_axis)
        ax.legend(frameon=True, fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(IMAGE_DIR / "04_knn_terhadap_k.png")
    plt.close(fig)

    # 5. Perbandingan seluruh model
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Validasi Ringkasan Akhir: Perbandingan Seluruh Model",
                 fontsize=16, fontweight="bold")
    warna_model = ["#C73E1D" if m == best_name else "#2E86AB"
                   for m in results_all["Model"]]
    ax = axes[0]
    bar_rmse = ax.bar(results_all["Model"], results_all["RMSE"],
                      color=warna_model, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(results_all)))
    ax.set_xticklabels(results_all["Model"], rotation=28, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Seluruh Model")
    ax.bar_label(bar_rmse, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    bar_r2 = ax.bar(results_all["Model"], results_all["R2"],
                    color=warna_model, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(results_all)))
    ax.set_xticklabels(results_all["Model"], rotation=28, ha="right")
    ax.set_ylabel("R2")
    ax.set_ylim(0, 1.0)
    ax.set_title("R2 Seluruh Model")
    ax.bar_label(bar_r2, fmt="%.4f", padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMAGE_DIR / "05_perbandingan_model.png")
    plt.close(fig)

    # 6. Kualitas prediksi model terbaik
    residual_best = y_test.values - y_pred_best
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Validasi Soal 3: Model Terbaik dan Perbandingannya",
                 fontsize=16, fontweight="bold")
    ax = axes[0]
    x_metrik = np.arange(3)
    w_metrik = 0.36
    nilai_lin = [metrics_lin[m] for m in ["RMSE", "MAE", "MAPE (%)"]]
    nilai_best = [best_metrics[m] for m in ["RMSE", "MAE", "MAPE (%)"]]
    ax.bar(x_metrik - w_metrik / 2, nilai_lin, w_metrik, label="Regresi Linier",
           color="#2E86AB", edgecolor="black", linewidth=0.4)
    ax.bar(x_metrik + w_metrik / 2, nilai_best, w_metrik, label=best_name,
           color="#C73E1D", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x_metrik)
    ax.set_xticklabels(["RMSE", "MAE", "MAPE (%)"])
    ax.set_title("Perbandingan Metrik Kesalahan")
    ax.set_ylabel("Nilai")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for i, (v_lin, v_best) in enumerate(zip(nilai_lin, nilai_best)):
        ax.text(i - w_metrik / 2, v_lin, f"{v_lin:.2f}", ha="center",
                va="bottom", fontsize=8)
        ax.text(i + w_metrik / 2, v_best, f"{v_best:.2f}", ha="center",
                va="bottom", fontsize=8)

    ax = axes[1]
    ax.scatter(y_test, y_pred_best, alpha=0.75, s=45, color="#6A4C93",
               edgecolor="black", linewidth=0.4)
    batas_best = [min(float(y_test.min()), float(y_pred_best.min())),
                  max(float(y_test.max()), float(y_pred_best.max()))]
    ax.plot(batas_best, batas_best, "r--", linewidth=1.8,
            label="Garis identitas (y = x)")
    ax.set_title(f"Aktual dan Prediksi: {best_name}")
    ax.set_xlabel("Harga aktual per satuan area")
    ax.set_ylabel("Harga prediksi per satuan area")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    sns.histplot(residual_best, bins=20, kde=True, color="#3A7D44", ax=ax)
    ax.axvline(0, color="#C73E1D", linestyle="--", linewidth=1.8)
    ax.set_title("Distribusi Residual Model Terbaik")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frekuensi")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMAGE_DIR / "06_model_terbaik.png")
    plt.close(fig)

    print("Grafik berhasil dibuat di direktori images/.")
    print(results_all.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
