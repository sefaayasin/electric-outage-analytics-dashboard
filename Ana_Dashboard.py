import os

import pandas as pd
import streamlit as st
import plotly.express as px


# =============================================================================
# SAYFA AYARI
# =============================================================================

st.set_page_config(
    page_title="Elektrik Dağıtım Kesinti Analitiği",
    page_icon="⚡",
    layout="wide"
)


# =============================================================================
# DOSYA YOLLARI
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

OUTAGES_PATH = os.path.join(DATA_DIR, "outages.csv")
OVERALL_KPIS_PATH = os.path.join(PROCESSED_DIR, "overall_kpis.csv")
REGION_KPIS_PATH = os.path.join(PROCESSED_DIR, "region_kpis.csv")
FEEDER_ANALYSIS_PATH = os.path.join(PROCESSED_DIR, "feeder_analysis.csv")
CAUSE_ANALYSIS_PATH = os.path.join(PROCESSED_DIR, "cause_analysis.csv")
SOURCE_ANALYSIS_PATH = os.path.join(PROCESSED_DIR, "source_analysis.csv")
NETWORK_ELEMENT_ANALYSIS_PATH = os.path.join(PROCESSED_DIR, "network_element_analysis.csv")
MONTHLY_TREND_PATH = os.path.join(PROCESSED_DIR, "monthly_trend.csv")
DISTRICT_RISK_SCORE_PATH = os.path.join(PROCESSED_DIR, "district_risk_score.csv")


# =============================================================================
# TÜRKÇE GÖRÜNÜM AYARLARI
# =============================================================================

COLUMN_NAME_TR = {
    "outage_id": "Kesinti ID",
    "city": "İl",
    "district": "İlçe",
    "feeder": "Fider",
    "transformer": "Trafo",
    "network_element_type": "Şebeke Elemanı Tipi",
    "outage_type": "Kesinti Tipi",
    "source": "Kaynak",
    "started_at": "Başlangıç Zamanı",
    "ended_at": "Bitiş Zamanı",
    "duration_min": "Kesinti Süresi (dk)",
    "affected_customer_count": "Etkilenen Müşteri Sayısı",
    "energy_not_supplied_kwh": "Verilemeyen Enerji (kWh)",
    "cause": "Kesinti Nedeni",
    "is_planned": "Planlı mı?",
    "is_force_majeure": "Mücbir Sebep mi?",
    "storm_flag": "Fırtına / Şiddetli Hava",
    "wind_speed": "Rüzgar Hızı",
    "precipitation_mm": "Yağış (mm)",
    "high_impact": "Yüksek Etkili mi?",
    "date": "Tarih",
    "year": "Yıl",
    "month": "Ay",
    "year_month": "Yıl / Ay",

    "outage_count": "Kesinti Sayısı",
    "planned_outage_count": "Planlı Kesinti Sayısı",
    "unplanned_outage_count": "Plansız Kesinti Sayısı",
    "force_majeure_count": "Mücbir Kesinti Sayısı",
    "high_impact_count": "Yüksek Etkili Kesinti Sayısı",
    "avg_duration_min": "Ortalama Süre (dk)",
    "max_duration_min": "Maksimum Süre (dk)",
    "total_affected_customer": "Toplam Etkilenen Müşteri",
    "avg_affected_customer": "Ortalama Etkilenen Müşteri",
    "total_ens_kwh": "Toplam ENS (kWh)",
    "total_customer_count": "Toplam Müşteri Sayısı",
    "residential_customer_count": "Mesken Müşteri Sayısı",
    "commercial_customer_count": "Ticarethane Müşteri Sayısı",
    "industrial_customer_count": "Sanayi Müşteri Sayısı",
    "customer_interruption_duration": "Müşteri Kesinti Süresi",
    "saidi": "SAIDI",
    "saifi": "SAIFI",
    "caidi": "CAIDI",
    "district_risk_score": "İlçe Risk Skoru",
    "risk_score": "Risk Skoru"
}

VALUE_NAME_TR = {
    "planned": "Planlı",
    "unplanned": "Plansız",
    0: "Hayır",
    1: "Evet"
}


def to_turkish_dataframe(df):
    """
    Sadece ekranda gösterim için kolon adlarını ve bazı değerleri Türkçeleştirir.
    Orijinal dataframe'i bozmaz.
    """
    if df is None or df.empty:
        return df

    display_df = df.copy()

    if "outage_type" in display_df.columns:
        display_df["outage_type"] = display_df["outage_type"].replace(VALUE_NAME_TR)

    for bool_col in ["is_planned", "is_force_majeure", "storm_flag", "high_impact"]:
        if bool_col in display_df.columns:
            display_df[bool_col] = display_df[bool_col].replace(VALUE_NAME_TR)

    display_df = display_df.rename(columns=COLUMN_NAME_TR)

    return display_df


def translate_outage_type(value):
    return VALUE_NAME_TR.get(value, value)


# =============================================================================
# VERİ OKUMA
# =============================================================================

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_data():
    outages_df = load_csv(OUTAGES_PATH)
    overall_kpis_df = load_csv(OVERALL_KPIS_PATH)
    region_kpis_df = load_csv(REGION_KPIS_PATH)
    feeder_analysis_df = load_csv(FEEDER_ANALYSIS_PATH)
    cause_analysis_df = load_csv(CAUSE_ANALYSIS_PATH)
    source_analysis_df = load_csv(SOURCE_ANALYSIS_PATH)
    network_element_analysis_df = load_csv(NETWORK_ELEMENT_ANALYSIS_PATH)
    monthly_trend_df = load_csv(MONTHLY_TREND_PATH)
    district_risk_score_df = load_csv(DISTRICT_RISK_SCORE_PATH)

    if not outages_df.empty:
        outages_df["started_at"] = pd.to_datetime(outages_df["started_at"])
        outages_df["ended_at"] = pd.to_datetime(outages_df["ended_at"])
        outages_df["year"] = outages_df["started_at"].dt.year
        outages_df["month"] = outages_df["started_at"].dt.month
        outages_df["year_month"] = outages_df["started_at"].dt.to_period("M").astype(str)

    return {
        "outages": outages_df,
        "overall_kpis": overall_kpis_df,
        "region_kpis": region_kpis_df,
        "feeder_analysis": feeder_analysis_df,
        "cause_analysis": cause_analysis_df,
        "source_analysis": source_analysis_df,
        "network_element_analysis": network_element_analysis_df,
        "monthly_trend": monthly_trend_df,
        "district_risk_score": district_risk_score_df
    }


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def format_number(value, decimals=0):
    try:
        if decimals == 0:
            return f"{float(value):,.0f}".replace(",", ".")
        return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return value


def check_required_files():
    required_paths = [
        OUTAGES_PATH,
        OVERALL_KPIS_PATH,
        REGION_KPIS_PATH,
        FEEDER_ANALYSIS_PATH,
        CAUSE_ANALYSIS_PATH,
        SOURCE_ANALYSIS_PATH,
        NETWORK_ELEMENT_ANALYSIS_PATH,
        MONTHLY_TREND_PATH,
        DISTRICT_RISK_SCORE_PATH
    ]

    missing_files = [path for path in required_paths if not os.path.exists(path)]

    if missing_files:
        st.error("Bazı veri dosyaları bulunamadı.")
        st.write("Önce aşağıdaki komutları sırasıyla çalıştırmalısın:")

        st.code(
            """
python src/generate_data.py
python src/analysis.py
streamlit run Ana_Dashboard.py
            """,
            language="bash"
        )

        st.write("Eksik dosyalar:")
        for file_path in missing_files:
            st.write(f"- {file_path}")

        st.stop()


def apply_filters(outages_df, selected_city, selected_district, selected_outage_type, selected_source):
    filtered_df = outages_df.copy()

    if selected_city != "Tümü":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    if selected_district != "Tümü":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]

    if selected_outage_type != "Tümü":
        filtered_df = filtered_df[filtered_df["outage_type"] == selected_outage_type]

    if selected_source != "Tümü":
        filtered_df = filtered_df[filtered_df["source"] == selected_source]

    return filtered_df


def calculate_filtered_kpis(filtered_df):
    if filtered_df.empty:
        return {
            "total_outage_count": 0,
            "planned_outage_count": 0,
            "unplanned_outage_count": 0,
            "high_impact_count": 0,
            "force_majeure_count": 0,
            "total_ens_kwh": 0,
            "avg_duration_min": 0,
            "avg_affected_customer": 0
        }

    return {
        "total_outage_count": len(filtered_df),
        "planned_outage_count": int(filtered_df["is_planned"].sum()),
        "unplanned_outage_count": int(len(filtered_df) - filtered_df["is_planned"].sum()),
        "high_impact_count": int(filtered_df["high_impact"].sum()),
        "force_majeure_count": int(filtered_df["is_force_majeure"].sum()),
        "total_ens_kwh": float(filtered_df["energy_not_supplied_kwh"].sum()),
        "avg_duration_min": float(filtered_df["duration_min"].mean()),
        "avg_affected_customer": float(filtered_df["affected_customer_count"].mean())
    }


def filter_summary_tables(df, selected_city, selected_district):
    if df.empty:
        return df

    filtered_df = df.copy()

    if "city" in filtered_df.columns and selected_city != "Tümü":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    if "district" in filtered_df.columns and selected_district != "Tümü":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]

    return filtered_df


# =============================================================================
# ANA UYGULAMA
# =============================================================================

check_required_files()
data = load_data()

outages_df = data["outages"]
overall_kpis_df = data["overall_kpis"]
region_kpis_df = data["region_kpis"]
feeder_analysis_df = data["feeder_analysis"]
cause_analysis_df = data["cause_analysis"]
source_analysis_df = data["source_analysis"]
network_element_analysis_df = data["network_element_analysis"]
monthly_trend_df = data["monthly_trend"]
district_risk_score_df = data["district_risk_score"]


# =============================================================================
# BAŞLIK
# =============================================================================

st.title("⚡ Elektrik Dağıtım Kesinti Analitiği ve Risk Dashboard’u")

st.markdown(
    """
Bu dashboard; kesinti kayıtları üzerinden **KPI takibi, bölgesel risk analizi, fider performansı,
kesinti nedeni analizi ve operasyonel karar destek** amacıyla hazırlanmıştır.
"""
)


# =============================================================================
# SIDEBAR FİLTRELER
# =============================================================================

st.sidebar.header("Filtreler")

city_options = ["Tümü"] + sorted(outages_df["city"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("İl", city_options)

if selected_city == "Tümü":
    district_options = ["Tümü"] + sorted(outages_df["district"].dropna().unique().tolist())
else:
    district_options = ["Tümü"] + sorted(
        outages_df[outages_df["city"] == selected_city]["district"].dropna().unique().tolist()
    )

selected_district = st.sidebar.selectbox("İlçe", district_options)

outage_type_raw_options = sorted(outages_df["outage_type"].dropna().unique().tolist())
outage_type_options = ["Tümü"] + outage_type_raw_options

selected_outage_type_label = st.sidebar.selectbox(
    "Kesinti Tipi",
    outage_type_options,
    format_func=lambda x: translate_outage_type(x)
)

selected_outage_type = selected_outage_type_label

source_options = ["Tümü"] + sorted(outages_df["source"].dropna().unique().tolist())
selected_source = st.sidebar.selectbox("Kaynak", source_options)

st.sidebar.divider()

st.sidebar.write("Veri Aralığı")

min_date = outages_df["started_at"].min().date()
max_date = outages_df["started_at"].max().date()

selected_date_range = st.sidebar.date_input(
    "Başlangıç / Bitiş Tarihi",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered_outages_df = apply_filters(
    outages_df=outages_df,
    selected_city=selected_city,
    selected_district=selected_district,
    selected_outage_type=selected_outage_type,
    selected_source=selected_source
)

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range

    filtered_outages_df = filtered_outages_df[
        (filtered_outages_df["started_at"].dt.date >= start_date)
        & (filtered_outages_df["started_at"].dt.date <= end_date)
    ]


# =============================================================================
# KPI KARTLARI
# =============================================================================

filtered_kpis = calculate_filtered_kpis(filtered_outages_df)

st.subheader("Genel KPI Özeti")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("Toplam Kesinti", format_number(filtered_kpis["total_outage_count"]))

with kpi_col2:
    st.metric("Planlı Kesinti", format_number(filtered_kpis["planned_outage_count"]))

with kpi_col3:
    st.metric("Plansız Kesinti", format_number(filtered_kpis["unplanned_outage_count"]))

with kpi_col4:
    st.metric("Yüksek Etkili Kesinti", format_number(filtered_kpis["high_impact_count"]))

kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)

with kpi_col5:
    st.metric("Mücbir Kesinti", format_number(filtered_kpis["force_majeure_count"]))

with kpi_col6:
    st.metric("Toplam ENS kWh", format_number(filtered_kpis["total_ens_kwh"], 2))

with kpi_col7:
    st.metric("Ort. Süre dk", format_number(filtered_kpis["avg_duration_min"], 2))

with kpi_col8:
    st.metric("Ort. Etkilenen Müşteri", format_number(filtered_kpis["avg_affected_customer"], 2))


# =============================================================================
# GENEL SAIDI / SAIFI / CAIDI
# =============================================================================

if not overall_kpis_df.empty:
    st.subheader("Dağıtım KPI Değerleri")

    overall = overall_kpis_df.iloc[0]

    saidi_col, saifi_col, caidi_col, ens_col = st.columns(4)

    with saidi_col:
        st.metric("SAIDI dk/müşteri", format_number(overall["saidi_min_per_customer"], 4))

    with saifi_col:
        st.metric("SAIFI kesinti/müşteri", format_number(overall["saifi_interruption_per_customer"], 4))

    with caidi_col:
        st.metric("CAIDI dk/kesinti", format_number(overall["caidi_min_per_interruption"], 4))

    with ens_col:
        st.metric("Genel ENS kWh", format_number(overall["total_ens_kwh"], 2))


# =============================================================================
# GRAFİKLER
# =============================================================================

st.subheader("Kesinti Trendleri")

if filtered_outages_df.empty:
    st.warning("Seçilen filtrelere uygun veri bulunamadı.")
else:
    monthly_filtered = filtered_outages_df.groupby("year_month", as_index=False).agg(
        outage_count=("outage_id", "count"),
        total_ens_kwh=("energy_not_supplied_kwh", "sum"),
        avg_duration_min=("duration_min", "mean"),
        high_impact_count=("high_impact", "sum")
    )

    monthly_filtered["avg_duration_min"] = monthly_filtered["avg_duration_min"].round(2)
    monthly_filtered["total_ens_kwh"] = monthly_filtered["total_ens_kwh"].round(2)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_monthly_count = px.line(
            monthly_filtered,
            x="year_month",
            y="outage_count",
            markers=True,
            title="Aylık Kesinti Sayısı",
            labels={
                "year_month": "Ay",
                "outage_count": "Kesinti Sayısı"
            }
        )
        st.plotly_chart(fig_monthly_count, use_container_width=True)

    with chart_col2:
        fig_monthly_ens = px.bar(
            monthly_filtered,
            x="year_month",
            y="total_ens_kwh",
            title="Aylık Verilemeyen Enerji",
            labels={
                "year_month": "Ay",
                "total_ens_kwh": "ENS kWh"
            }
        )
        st.plotly_chart(fig_monthly_ens, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        outage_type_dist = filtered_outages_df.groupby("outage_type", as_index=False).agg(
            outage_count=("outage_id", "count")
        )

        outage_type_dist["Kesinti Tipi"] = outage_type_dist["outage_type"].replace(VALUE_NAME_TR)

        fig_type = px.pie(
            outage_type_dist,
            names="Kesinti Tipi",
            values="outage_count",
            title="Planlı / Plansız Kesinti Dağılımı",
            labels={
                "outage_count": "Kesinti Sayısı"
            }
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with chart_col4:
        source_dist = filtered_outages_df.groupby("source", as_index=False).agg(
            outage_count=("outage_id", "count")
        )

        fig_source = px.bar(
            source_dist,
            x="source",
            y="outage_count",
            title="Kaynak Bazlı Kesinti Sayısı",
            labels={
                "source": "Kaynak",
                "outage_count": "Kesinti Sayısı"
            }
        )
        st.plotly_chart(fig_source, use_container_width=True)


# =============================================================================
# RİSK ANALİZİ
# =============================================================================

st.subheader("Bölgesel Risk Analizi")

filtered_district_risk_df = filter_summary_tables(
    district_risk_score_df,
    selected_city,
    selected_district
)

filtered_region_kpis_df = filter_summary_tables(
    region_kpis_df,
    selected_city,
    selected_district
)

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    top_districts = filtered_district_risk_df.head(10)

    if not top_districts.empty:
        fig_risk = px.bar(
            top_districts.sort_values("district_risk_score"),
            x="district_risk_score",
            y="district",
            orientation="h",
            title="En Riskli 10 İlçe",
            labels={
                "district_risk_score": "Risk Skoru",
                "district": "İlçe"
            },
            hover_data=["city", "outage_count", "high_impact_count", "saidi", "saifi"]
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Risk skoru için veri bulunamadı.")

with risk_col2:
    if not filtered_region_kpis_df.empty:
        top_saidi = filtered_region_kpis_df.sort_values("saidi", ascending=False).head(10)

        fig_saidi = px.bar(
            top_saidi.sort_values("saidi"),
            x="saidi",
            y="district",
            orientation="h",
            title="SAIDI Değeri En Yüksek 10 İlçe",
            labels={
                "saidi": "SAIDI",
                "district": "İlçe"
            },
            hover_data=["city", "outage_count", "total_ens_kwh", "caidi"]
        )
        st.plotly_chart(fig_saidi, use_container_width=True)
    else:
        st.info("SAIDI analizi için veri bulunamadı.")


# =============================================================================
# FİDER ANALİZİ
# =============================================================================

st.subheader("Fider Bazlı Risk Analizi")

filtered_feeder_df = filter_summary_tables(
    feeder_analysis_df,
    selected_city,
    selected_district
)

top_feeders = filtered_feeder_df.head(15)

if not top_feeders.empty:
    fig_feeder = px.bar(
        top_feeders.sort_values("risk_score"),
        x="risk_score",
        y="feeder",
        orientation="h",
        title="En Riskli 15 Fider",
        labels={
            "risk_score": "Risk Skoru",
            "feeder": "Fider"
        },
        hover_data=[
            "city",
            "district",
            "outage_count",
            "high_impact_count",
            "avg_duration_min",
            "total_affected_customer",
            "total_ens_kwh"
        ]
    )

    st.plotly_chart(fig_feeder, use_container_width=True)

    st.dataframe(
        to_turkish_dataframe(
            top_feeders[
                [
                    "city",
                    "district",
                    "feeder",
                    "risk_score",
                    "outage_count",
                    "high_impact_count",
                    "avg_duration_min",
                    "total_affected_customer",
                    "total_ens_kwh",
                    "force_majeure_count"
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Fider analizi için veri bulunamadı.")


# =============================================================================
# KESİNTİ NEDENİ / KAYNAK / ŞEBEKE ELEMANI
# =============================================================================

st.subheader("Operasyonel Dağılım Analizleri")

op_col1, op_col2 = st.columns(2)

with op_col1:
    if not filtered_outages_df.empty:
        cause_filtered = filtered_outages_df.groupby("cause", as_index=False).agg(
            outage_count=("outage_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            total_ens_kwh=("energy_not_supplied_kwh", "sum"),
            high_impact_count=("high_impact", "sum")
        )

        cause_filtered["avg_duration_min"] = cause_filtered["avg_duration_min"].round(2)
        cause_filtered["total_ens_kwh"] = cause_filtered["total_ens_kwh"].round(2)

        fig_cause = px.bar(
            cause_filtered.sort_values("outage_count", ascending=True),
            x="outage_count",
            y="cause",
            orientation="h",
            title="Kesinti Nedeni Bazlı Dağılım",
            labels={
                "outage_count": "Kesinti Sayısı",
                "cause": "Kesinti Nedeni"
            },
            hover_data=["avg_duration_min", "total_ens_kwh", "high_impact_count"]
        )
        st.plotly_chart(fig_cause, use_container_width=True)
    else:
        st.info("Kesinti nedeni analizi için veri bulunamadı.")

with op_col2:
    if not filtered_outages_df.empty:
        network_filtered = filtered_outages_df.groupby("network_element_type", as_index=False).agg(
            outage_count=("outage_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            total_ens_kwh=("energy_not_supplied_kwh", "sum"),
            high_impact_count=("high_impact", "sum")
        )

        network_filtered["avg_duration_min"] = network_filtered["avg_duration_min"].round(2)
        network_filtered["total_ens_kwh"] = network_filtered["total_ens_kwh"].round(2)

        fig_network = px.bar(
            network_filtered.sort_values("outage_count", ascending=True),
            x="outage_count",
            y="network_element_type",
            orientation="h",
            title="Şebeke Elemanı Bazlı Kesinti Dağılımı",
            labels={
                "outage_count": "Kesinti Sayısı",
                "network_element_type": "Şebeke Elemanı"
            },
            hover_data=["avg_duration_min", "total_ens_kwh", "high_impact_count"]
        )
        st.plotly_chart(fig_network, use_container_width=True)
    else:
        st.info("Şebeke elemanı analizi için veri bulunamadı.")


# =============================================================================
# DETAY TABLOLAR
# =============================================================================

st.subheader("Detay Analiz Tabloları")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "İlçe Risk Skoru",
        "İl / İlçe KPI",
        "Fider Analizi",
        "Kesinti Nedenleri",
        "Ham Kesinti Verisi"
    ]
)

with tab1:
    st.dataframe(
        to_turkish_dataframe(filtered_district_risk_df),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.dataframe(
        to_turkish_dataframe(filtered_region_kpis_df),
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.dataframe(
        to_turkish_dataframe(filtered_feeder_df),
        use_container_width=True,
        hide_index=True
    )

with tab4:
    if not filtered_outages_df.empty:
        cause_table = filtered_outages_df.groupby("cause", as_index=False).agg(
            outage_count=("outage_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            total_affected_customer=("affected_customer_count", "sum"),
            total_ens_kwh=("energy_not_supplied_kwh", "sum"),
            high_impact_count=("high_impact", "sum")
        )

        cause_table["avg_duration_min"] = cause_table["avg_duration_min"].round(2)
        cause_table["total_ens_kwh"] = cause_table["total_ens_kwh"].round(2)

        st.dataframe(
            to_turkish_dataframe(cause_table.sort_values("outage_count", ascending=False)),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Kesinti nedeni tablosu için veri bulunamadı.")

with tab5:
    st.dataframe(
        to_turkish_dataframe(filtered_outages_df),
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# DOSYA İNDİRME
# =============================================================================

st.subheader("Çıktı Alma")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="Filtrelenmiş Kesinti Verisini CSV İndir",
        data=to_turkish_dataframe(filtered_outages_df).to_csv(index=False, encoding="utf-8-sig"),
        file_name="filtrelenmis_kesinti_verisi.csv",
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="İlçe Risk Skorunu CSV İndir",
        data=to_turkish_dataframe(filtered_district_risk_df).to_csv(index=False, encoding="utf-8-sig"),
        file_name="ilce_risk_skoru.csv",
        mime="text/csv"
    )

with download_col3:
    st.download_button(
        label="Fider Analizini CSV İndir",
        data=to_turkish_dataframe(filtered_feeder_df).to_csv(index=False, encoding="utf-8-sig"),
        file_name="fider_analizi.csv",
        mime="text/csv"
    )