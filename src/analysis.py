import os
import pandas as pd
import numpy as np


# =============================================================================
# DOSYA YOLLARI
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTAGES_PATH = os.path.join(DATA_DIR, "outages.csv")
CUSTOMER_REGION_PATH = os.path.join(DATA_DIR, "customer_region.csv")


# =============================================================================
# VERİ OKUMA
# =============================================================================

def load_data():
    """
    Outage ve müşteri bölge verilerini okur.
    """

    if not os.path.exists(OUTAGES_PATH):
        raise FileNotFoundError(
            f"{OUTAGES_PATH} bulunamadı. Önce python src/generate_data.py çalıştırmalısın."
        )

    if not os.path.exists(CUSTOMER_REGION_PATH):
        raise FileNotFoundError(
            f"{CUSTOMER_REGION_PATH} bulunamadı. Önce python src/generate_data.py çalıştırmalısın."
        )

    outages_df = pd.read_csv(OUTAGES_PATH)
    customer_region_df = pd.read_csv(CUSTOMER_REGION_PATH)

    outages_df["started_at"] = pd.to_datetime(outages_df["started_at"])
    outages_df["ended_at"] = pd.to_datetime(outages_df["ended_at"])

    outages_df["date"] = outages_df["started_at"].dt.date
    outages_df["year"] = outages_df["started_at"].dt.year
    outages_df["month"] = outages_df["started_at"].dt.month
    outages_df["year_month"] = outages_df["started_at"].dt.to_period("M").astype(str)

    return outages_df, customer_region_df


# =============================================================================
# GENEL KPI HESAPLAMA
# =============================================================================

def calculate_overall_kpis(outages_df, customer_region_df):
    """
    Genel dağıtım KPI değerlerini hesaplar.
    """

    total_customer_count = customer_region_df["total_customer_count"].sum()

    total_outage_count = len(outages_df)

    total_customer_interruption_duration = (
        outages_df["duration_min"] * outages_df["affected_customer_count"]
    ).sum()

    total_customer_interruption_count = outages_df["affected_customer_count"].sum()

    total_ens = outages_df["energy_not_supplied_kwh"].sum()

    saidi = total_customer_interruption_duration / total_customer_count
    saifi = total_customer_interruption_count / total_customer_count
    caidi = saidi / saifi if saifi != 0 else 0

    planned_outage_count = outages_df[outages_df["is_planned"] == 1].shape[0]
    unplanned_outage_count = outages_df[outages_df["is_planned"] == 0].shape[0]

    high_impact_count = outages_df[outages_df["high_impact"] == 1].shape[0]

    force_majeure_count = outages_df[outages_df["is_force_majeure"] == 1].shape[0]

    avg_duration = outages_df["duration_min"].mean()
    avg_affected_customer = outages_df["affected_customer_count"].mean()

    kpi_data = {
        "total_customer_count": total_customer_count,
        "total_outage_count": total_outage_count,
        "planned_outage_count": planned_outage_count,
        "unplanned_outage_count": unplanned_outage_count,
        "high_impact_count": high_impact_count,
        "force_majeure_count": force_majeure_count,
        "total_ens_kwh": round(total_ens, 2),
        "avg_duration_min": round(avg_duration, 2),
        "avg_affected_customer": round(avg_affected_customer, 2),
        "saidi_min_per_customer": round(saidi, 4),
        "saifi_interruption_per_customer": round(saifi, 4),
        "caidi_min_per_interruption": round(caidi, 4)
    }

    return pd.DataFrame([kpi_data])


# =============================================================================
# İL / İLÇE BAZLI KPI
# =============================================================================

def calculate_region_kpis(outages_df, customer_region_df):
    """
    İl ve ilçe bazlı SAIDI, SAIFI, CAIDI, ENS ve kesinti metriklerini hesaplar.
    """

    grouped = outages_df.groupby(["city", "district"], as_index=False).agg(
        outage_count=("outage_id", "count"),
        planned_outage_count=("is_planned", "sum"),
        force_majeure_count=("is_force_majeure", "sum"),
        high_impact_count=("high_impact", "sum"),
        avg_duration_min=("duration_min", "mean"),
        max_duration_min=("duration_min", "max"),
        total_affected_customer=("affected_customer_count", "sum"),
        avg_affected_customer=("affected_customer_count", "mean"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum")
    )

    grouped["unplanned_outage_count"] = grouped["outage_count"] - grouped["planned_outage_count"]

    grouped = grouped.merge(
        customer_region_df,
        on=["city", "district"],
        how="left"
    )

    grouped["customer_interruption_duration"] = (
        grouped["avg_duration_min"] * grouped["total_affected_customer"]
    )

    grouped["saidi"] = (
        grouped["customer_interruption_duration"] / grouped["total_customer_count"]
    )

    grouped["saifi"] = (
        grouped["total_affected_customer"] / grouped["total_customer_count"]
    )

    grouped["caidi"] = grouped.apply(
        lambda row: row["saidi"] / row["saifi"] if row["saifi"] != 0 else 0,
        axis=1
    )

    numeric_cols = [
        "avg_duration_min",
        "avg_affected_customer",
        "total_ens_kwh",
        "saidi",
        "saifi",
        "caidi"
    ]

    for col in numeric_cols:
        grouped[col] = grouped[col].round(4)

    return grouped.sort_values("saidi", ascending=False)


# =============================================================================
# FİDER BAZLI ANALİZ
# =============================================================================

def calculate_feeder_analysis(outages_df):
    """
    Fider bazlı kesinti performansını hesaplar.
    """

    feeder_df = outages_df.groupby(
        ["city", "district", "feeder"],
        as_index=False
    ).agg(
        outage_count=("outage_id", "count"),
        high_impact_count=("high_impact", "sum"),
        avg_duration_min=("duration_min", "mean"),
        max_duration_min=("duration_min", "max"),
        total_affected_customer=("affected_customer_count", "sum"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum"),
        force_majeure_count=("is_force_majeure", "sum")
    )

    feeder_df["avg_duration_min"] = feeder_df["avg_duration_min"].round(2)
    feeder_df["total_ens_kwh"] = feeder_df["total_ens_kwh"].round(2)

    feeder_df["risk_score"] = (
        feeder_df["outage_count"] * 0.25
        + feeder_df["high_impact_count"] * 1.50
        + feeder_df["avg_duration_min"] * 0.03
        + feeder_df["total_affected_customer"] * 0.002
        + feeder_df["total_ens_kwh"] * 0.0005
    )

    feeder_df["risk_score"] = feeder_df["risk_score"].round(2)

    return feeder_df.sort_values("risk_score", ascending=False)


# =============================================================================
# NEDEN / KAYNAK / ŞEBEKE ELEMANI ANALİZLERİ
# =============================================================================

def calculate_cause_analysis(outages_df):
    """
    Kesinti nedeni bazlı analiz üretir.
    """

    cause_df = outages_df.groupby("cause", as_index=False).agg(
        outage_count=("outage_id", "count"),
        avg_duration_min=("duration_min", "mean"),
        total_affected_customer=("affected_customer_count", "sum"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum"),
        high_impact_count=("high_impact", "sum")
    )

    cause_df["avg_duration_min"] = cause_df["avg_duration_min"].round(2)
    cause_df["total_ens_kwh"] = cause_df["total_ens_kwh"].round(2)

    return cause_df.sort_values("outage_count", ascending=False)


def calculate_source_analysis(outages_df):
    """
    Kesinti kaynağı bazlı analiz üretir.
    """

    source_df = outages_df.groupby("source", as_index=False).agg(
        outage_count=("outage_id", "count"),
        avg_duration_min=("duration_min", "mean"),
        total_affected_customer=("affected_customer_count", "sum"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum"),
        high_impact_count=("high_impact", "sum")
    )

    source_df["avg_duration_min"] = source_df["avg_duration_min"].round(2)
    source_df["total_ens_kwh"] = source_df["total_ens_kwh"].round(2)

    return source_df.sort_values("outage_count", ascending=False)


def calculate_network_element_analysis(outages_df):
    """
    Şebeke elemanı tipine göre analiz üretir.
    """

    network_df = outages_df.groupby("network_element_type", as_index=False).agg(
        outage_count=("outage_id", "count"),
        avg_duration_min=("duration_min", "mean"),
        total_affected_customer=("affected_customer_count", "sum"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum"),
        high_impact_count=("high_impact", "sum")
    )

    network_df["avg_duration_min"] = network_df["avg_duration_min"].round(2)
    network_df["total_ens_kwh"] = network_df["total_ens_kwh"].round(2)

    return network_df.sort_values("outage_count", ascending=False)


# =============================================================================
# AYLIK TREND ANALİZİ
# =============================================================================

def calculate_monthly_trend(outages_df):
    """
    Aylık kesinti trendini hesaplar.
    """

    monthly_df = outages_df.groupby("year_month", as_index=False).agg(
        outage_count=("outage_id", "count"),
        planned_outage_count=("is_planned", "sum"),
        high_impact_count=("high_impact", "sum"),
        avg_duration_min=("duration_min", "mean"),
        total_affected_customer=("affected_customer_count", "sum"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum")
    )

    monthly_df["unplanned_outage_count"] = (
        monthly_df["outage_count"] - monthly_df["planned_outage_count"]
    )

    monthly_df["avg_duration_min"] = monthly_df["avg_duration_min"].round(2)
    monthly_df["total_ens_kwh"] = monthly_df["total_ens_kwh"].round(2)

    return monthly_df.sort_values("year_month")


# =============================================================================
# RİSKLİ İLÇE SKORU
# =============================================================================

def calculate_district_risk_score(region_kpi_df):
    """
    İlçe bazlı risk skoru üretir.
    """

    df = region_kpi_df.copy()

    score_columns = [
        "outage_count",
        "unplanned_outage_count",
        "high_impact_count",
        "avg_duration_min",
        "total_affected_customer",
        "total_ens_kwh",
        "saidi",
        "saifi"
    ]

    for col in score_columns:
        min_val = df[col].min()
        max_val = df[col].max()

        if max_val == min_val:
            df[f"{col}_norm"] = 0
        else:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)

    df["district_risk_score"] = (
        df["outage_count_norm"] * 0.12
        + df["unplanned_outage_count_norm"] * 0.14
        + df["high_impact_count_norm"] * 0.16
        + df["avg_duration_min_norm"] * 0.10
        + df["total_affected_customer_norm"] * 0.14
        + df["total_ens_kwh_norm"] * 0.14
        + df["saidi_norm"] * 0.12
        + df["saifi_norm"] * 0.08
    ) * 100

    df["district_risk_score"] = df["district_risk_score"].round(2)

    output_columns = [
        "city",
        "district",
        "district_risk_score",
        "outage_count",
        "unplanned_outage_count",
        "high_impact_count",
        "avg_duration_min",
        "total_affected_customer",
        "total_ens_kwh",
        "saidi",
        "saifi",
        "caidi"
    ]

    return df[output_columns].sort_values("district_risk_score", ascending=False)


# =============================================================================
# ÇIKTILARI KAYDETME
# =============================================================================

def save_outputs(outputs: dict):
    """
    Tüm analiz çıktılarını CSV olarak kaydeder.
    """

    for name, df in outputs.items():
        output_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Kaydedildi: {output_path}")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    print("Analiz başladı...")

    outages_df, customer_region_df = load_data()

    overall_kpis = calculate_overall_kpis(outages_df, customer_region_df)
    region_kpis = calculate_region_kpis(outages_df, customer_region_df)
    feeder_analysis = calculate_feeder_analysis(outages_df)
    cause_analysis = calculate_cause_analysis(outages_df)
    source_analysis = calculate_source_analysis(outages_df)
    network_element_analysis = calculate_network_element_analysis(outages_df)
    monthly_trend = calculate_monthly_trend(outages_df)
    district_risk_score = calculate_district_risk_score(region_kpis)

    outputs = {
        "overall_kpis": overall_kpis,
        "region_kpis": region_kpis,
        "feeder_analysis": feeder_analysis,
        "cause_analysis": cause_analysis,
        "source_analysis": source_analysis,
        "network_element_analysis": network_element_analysis,
        "monthly_trend": monthly_trend,
        "district_risk_score": district_risk_score
    }

    save_outputs(outputs)

    print("\nGenel KPI Özeti:")
    print(overall_kpis.T)

    print("\nEn riskli 10 ilçe:")
    print(district_risk_score.head(10))

    print("\nEn riskli 10 fider:")
    print(feeder_analysis.head(10))

    print("\nAnaliz tamamlandı.")


if __name__ == "__main__":
    main()