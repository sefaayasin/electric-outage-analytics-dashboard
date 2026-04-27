import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# SAYFA AYARI
# =============================================================================

st.set_page_config(
    page_title="Model Tahmin Ekranı",
    page_icon="🤖",
    layout="wide"
)


# =============================================================================
# DOSYA YOLLARI
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

OUTAGES_PATH = os.path.join(DATA_DIR, "outages.csv")
DURATION_MODEL_PATH = os.path.join(MODELS_DIR, "duration_model.pkl")
HIGH_IMPACT_MODEL_PATH = os.path.join(MODELS_DIR, "high_impact_model.pkl")


# =============================================================================
# VERİ / MODEL OKUMA
# =============================================================================

@st.cache_data
def load_outages():
    if not os.path.exists(OUTAGES_PATH):
        return pd.DataFrame()

    df = pd.read_csv(OUTAGES_PATH)
    return df


@st.cache_resource
def load_models():
    """
    Yayın ortamında hazır eğitilmiş model dosyalarını yükler.
    Kullanıcı her girişinde model eğitimi yapılmaz.
    """

    try:
        if not os.path.exists(DURATION_MODEL_PATH):
            return None, None, "duration_model.pkl bulunamadı."

        if not os.path.exists(HIGH_IMPACT_MODEL_PATH):
            return None, None, "high_impact_model.pkl bulunamadı."

        duration_model = joblib.load(DURATION_MODEL_PATH)
        high_impact_model = joblib.load(HIGH_IMPACT_MODEL_PATH)

        return duration_model, high_impact_model, None

    except Exception as error:
        return None, None, str(error)
            

def get_options(df, column_name, default_list):
    if df.empty or column_name not in df.columns:
        return default_list

    values = sorted(df[column_name].dropna().unique().tolist())

    if not values:
        return default_list

    return values


def build_prediction_dataframe(
    city,
    district,
    network_element_type,
    outage_type,
    source,
    cause,
    affected_customer_count,
    is_force_majeure,
    storm_flag,
    wind_speed,
    precipitation_mm,
    prediction_datetime
):
    is_planned = 1 if outage_type == "planned" else 0

    dayofweek = prediction_datetime.weekday()
    is_weekend = 1 if dayofweek in [5, 6] else 0

    input_data = {
        "city": city,
        "district": district,
        "network_element_type": network_element_type,
        "outage_type": outage_type,
        "source": source,
        "cause": cause,
        "affected_customer_count": affected_customer_count,
        "is_planned": is_planned,
        "is_force_majeure": is_force_majeure,
        "storm_flag": storm_flag,
        "wind_speed": wind_speed,
        "precipitation_mm": precipitation_mm,
        "year": prediction_datetime.year,
        "month": prediction_datetime.month,
        "day": prediction_datetime.day,
        "hour": prediction_datetime.hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend
    }

    return pd.DataFrame([input_data])


def calculate_estimated_ens(predicted_duration_min, affected_customer_count):
    """
    Tahmini ENS hesabı.
    Bu gerçek model çıktısı değil, kullanıcıya yaklaşık operasyonel etki göstermek için hesaplanır.
    """

    avg_kw_per_customer = 1.1
    estimated_ens = affected_customer_count * avg_kw_per_customer * (predicted_duration_min / 60)

    return round(estimated_ens, 2)


def get_risk_level(probability):
    if probability >= 0.75:
        return "Çok Yüksek Risk"
    elif probability >= 0.55:
        return "Yüksek Risk"
    elif probability >= 0.35:
        return "Orta Risk"
    else:
        return "Düşük Risk"


def get_operation_recommendation(probability, predicted_duration_min, affected_customer_count):
    recommendations = []

    if probability >= 0.75:
        recommendations.append("Kesinti yüksek etkili olma eğiliminde. Müdahale önceliği artırılmalı.")
    elif probability >= 0.55:
        recommendations.append("Kesinti için saha ekibi ve operasyon merkezi yakın takip yapmalı.")
    elif probability >= 0.35:
        recommendations.append("Kesinti orta riskli görünüyor. Standart müdahale planı izlenebilir.")
    else:
        recommendations.append("Kesinti düşük riskli görünüyor. Standart operasyon akışı yeterli olabilir.")

    if predicted_duration_min >= 180:
        recommendations.append("Tahmini süre 180 dakikanın üzerinde. Uzun kesinti takibi yapılmalı.")

    if affected_customer_count >= 1000:
        recommendations.append("Etkilenen müşteri sayısı yüksek. Müşteri bilgilendirme süreci kontrol edilmeli.")

    if affected_customer_count >= 3000:
        recommendations.append("Çok yüksek müşteri etkisi var. Çağrı merkezi, SMS ve saha koordinasyonu önceliklendirilmeli.")

    return recommendations


# =============================================================================
# ANA SAYFA
# =============================================================================

st.title("🤖 Kesinti Etki Tahmin Ekranı")

st.markdown(
    """
Bu ekran, geçmiş kesinti verileri üzerinden eğitilen modelleri kullanarak yeni bir kesinti senaryosu için:

- Tahmini kesinti süresini,
- Yüksek etkili kesinti olasılığını,
- Yaklaşık ENS etkisini,
- Operasyonel risk seviyesini

hesaplar.
"""
)

outages_df = load_outages()
duration_model, high_impact_model, model_error = load_models()

if outages_df.empty:
    st.error("outages.csv bulunamadı. Önce aşağıdaki komutu çalıştırmalısın:")
    st.code("python src/generate_data.py", language="bash")
    st.stop()

if duration_model is None or high_impact_model is None:
    st.error("Model dosyaları yüklenemedi veya yeniden eğitilemedi.")
    if model_error:
        st.code(model_error)
    st.stop()


# =============================================================================
# FORM ALANI
# =============================================================================

city_options = get_options(
    outages_df,
    "city",
    ["Ankara", "Konya", "Kayseri", "Eskişehir", "Sivas"]
)

network_element_options = get_options(
    outages_df,
    "network_element_type",
    ["Fider", "Trafo", "Kesici", "Ayırıcı", "Direk", "Kablo", "Dağıtım Merkezi", "Hücre"]
)

source_options = get_options(
    outages_df,
    "source",
    ["SCADA", "OSOS", "CRM", "Mobile", "Manual", "OMS"]
)

cause_options = get_options(
    outages_df,
    "cause",
    [
        "Hava muhalefeti",
        "Ekipman arızası",
        "Bakım çalışması",
        "Kablo arızası",
        "Trafo arızası",
        "Aşırı yüklenme",
        "Kuş / hayvan teması",
        "Üçüncü şahıs müdahalesi",
        "Ağaç teması",
        "Bilinmeyen"
    ]
)

st.subheader("Kesinti Senaryosu")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("İl", city_options)

        district_options = sorted(
            outages_df[outages_df["city"] == city]["district"].dropna().unique().tolist()
        )

        if not district_options:
            district_options = ["Merkez"]

        district = st.selectbox("İlçe", district_options)

        network_element_type = st.selectbox(
            "Şebeke Elemanı Tipi",
            network_element_options
        )

        outage_type = st.selectbox(
            "Kesinti Tipi",
            ["planned", "unplanned"]
        )

    with col2:
        source = st.selectbox("Kaynak", source_options)

        cause = st.selectbox("Kesinti Nedeni", cause_options)

        affected_customer_count = st.number_input(
            "Etkilenen Müşteri Sayısı",
            min_value=1,
            max_value=100000,
            value=1200,
            step=100
        )

        prediction_date = st.date_input(
            "Kesinti Tarihi",
            value=datetime.now().date()
        )

    with col3:
        prediction_hour = st.slider(
            "Kesinti Saati",
            min_value=0,
            max_value=23,
            value=datetime.now().hour
        )

        is_force_majeure = st.selectbox(
            "Mücbir Sebep",
            [0, 1],
            format_func=lambda x: "Evet" if x == 1 else "Hayır"
        )

        storm_flag = st.selectbox(
            "Fırtına / Şiddetli Hava",
            [0, 1],
            format_func=lambda x: "Evet" if x == 1 else "Hayır"
        )

        wind_speed = st.number_input(
            "Rüzgar Hızı",
            min_value=0.0,
            max_value=120.0,
            value=18.0,
            step=1.0
        )

        precipitation_mm = st.number_input(
            "Yağış mm",
            min_value=0.0,
            max_value=200.0,
            value=2.0,
            step=1.0
        )

    submitted = st.form_submit_button("Tahmin Et")


# =============================================================================
# TAHMİN
# =============================================================================

if submitted:
    prediction_datetime = datetime(
        year=prediction_date.year,
        month=prediction_date.month,
        day=prediction_date.day,
        hour=prediction_hour
    )

    input_df = build_prediction_dataframe(
        city=city,
        district=district,
        network_element_type=network_element_type,
        outage_type=outage_type,
        source=source,
        cause=cause,
        affected_customer_count=affected_customer_count,
        is_force_majeure=is_force_majeure,
        storm_flag=storm_flag,
        wind_speed=wind_speed,
        precipitation_mm=precipitation_mm,
        prediction_datetime=prediction_datetime
    )

    predicted_duration = float(duration_model.predict(input_df)[0])
    predicted_duration = max(1, predicted_duration)

    predicted_high_impact = int(high_impact_model.predict(input_df)[0])

    if hasattr(high_impact_model.named_steps["model"], "predict_proba"):
        high_impact_probability = float(high_impact_model.predict_proba(input_df)[0][1])
    else:
        high_impact_probability = 0.0

    estimated_ens = calculate_estimated_ens(
        predicted_duration_min=predicted_duration,
        affected_customer_count=affected_customer_count
    )

    risk_level = get_risk_level(high_impact_probability)

    recommendations = get_operation_recommendation(
        probability=high_impact_probability,
        predicted_duration_min=predicted_duration,
        affected_customer_count=affected_customer_count
    )

    st.divider()

    st.subheader("Tahmin Sonucu")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:
        st.metric(
            "Tahmini Kesinti Süresi",
            f"{predicted_duration:.0f} dk"
        )

    with result_col2:
        st.metric(
            "Yüksek Etki Olasılığı",
            f"%{high_impact_probability * 100:.1f}"
        )

    with result_col3:
        st.metric(
            "Tahmini ENS",
            f"{estimated_ens:,.2f} kWh".replace(",", ".")
        )

    with result_col4:
        st.metric(
            "Risk Seviyesi",
            risk_level
        )

    if predicted_high_impact == 1:
        st.warning("Model bu kesintiyi yüksek etkili kesinti olarak sınıflandırdı.")
    else:
        st.success("Model bu kesintiyi yüksek etkili olmayan kesinti olarak sınıflandırdı.")

    st.subheader("Operasyonel Öneriler")

    for item in recommendations:
        st.write(f"- {item}")

    st.subheader("Modele Gönderilen Girdi")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )
