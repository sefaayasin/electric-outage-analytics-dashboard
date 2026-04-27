import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# =============================================================================
# AYARLAR
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2026, 4, 27)

OUTAGE_COUNT = 15000


# =============================================================================
# SABİT VERİLER
# =============================================================================

CITIES_DISTRICTS = {
    "Ankara": [
        "Çankaya", "Keçiören", "Mamak", "Yenimahalle", "Etimesgut",
        "Sincan", "Altındağ", "Gölbaşı", "Pursaklar", "Polatlı"
    ],
    "Konya": [
        "Selçuklu", "Meram", "Karatay", "Ereğli", "Akşehir",
        "Beyşehir", "Cihanbeyli", "Ilgın", "Seydişehir", "Kulu"
    ],
    "Kayseri": [
        "Melikgazi", "Kocasinan", "Talas", "Develi", "Yahyalı",
        "Bünyan", "İncesu", "Pınarbaşı", "Yeşilhisar", "Tomarza"
    ],
    "Eskişehir": [
        "Odunpazarı", "Tepebaşı", "Sivrihisar", "Çifteler", "Alpu",
        "Beylikova", "İnönü", "Mihalıççık", "Seyitgazi", "Mahmudiye"
    ],
    "Sivas": [
        "Merkez", "Şarkışla", "Suşehri", "Yıldızeli", "Zara",
        "Kangal", "Gemerek", "Divriği", "Gürün", "Hafik"
    ]
}

NETWORK_ELEMENT_TYPES = [
    "Fider",
    "Trafo",
    "Kesici",
    "Ayırıcı",
    "Direk",
    "Kablo",
    "Dağıtım Merkezi",
    "Hücre"
]

OUTAGE_TYPES = [
    "planned",
    "unplanned"
]

SOURCES = [
    "SCADA",
    "OSOS",
    "CRM",
    "Mobile",
    "Manual",
    "OMS"
]

CAUSES = [
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

FORCE_MAJEURE_CAUSES = [
    "Hava muhalefeti",
    "Üçüncü şahıs müdahalesi"
]


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def random_date(start_date: datetime, end_date: datetime) -> datetime:
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


def generate_feeder(city: str, district: str) -> str:
    city_code = city[:3].upper().replace("İ", "I").replace("Ş", "S")
    district_code = district[:3].upper().replace("İ", "I").replace("Ş", "S")
    number = random.randint(1, 60)
    return f"{city_code}-{district_code}-FDR-{number:03d}"


def generate_transformer(city: str, district: str) -> str:
    city_code = city[:3].upper().replace("İ", "I").replace("Ş", "S")
    district_code = district[:3].upper().replace("İ", "I").replace("Ş", "S")
    number = random.randint(1, 500)
    return f"{city_code}-{district_code}-TR-{number:04d}"


def calculate_duration_minutes(outage_type: str, cause: str) -> int:
    """
    Planlı kesintiler daha uzun ama daha kontrollü,
    plansız kesintiler ise daha değişken olacak şekilde üretilir.
    """

    if outage_type == "planned":
        duration = np.random.normal(loc=240, scale=90)
    else:
        duration = np.random.exponential(scale=95)

    if cause == "Hava muhalefeti":
        duration *= np.random.uniform(1.4, 2.4)
    elif cause == "Trafo arızası":
        duration *= np.random.uniform(1.2, 1.8)
    elif cause == "Kablo arızası":
        duration *= np.random.uniform(1.2, 2.0)
    elif cause == "Bakım çalışması":
        duration *= np.random.uniform(0.9, 1.4)

    duration = max(5, int(duration))
    duration = min(duration, 1440)

    return duration


def calculate_affected_customers(network_element_type: str, outage_type: str) -> int:
    """
    Fider ve dağıtım merkezi gibi elemanlar daha fazla müşteriyi etkiler.
    """

    base_ranges = {
        "Fider": (300, 4500),
        "Trafo": (30, 800),
        "Kesici": (100, 2500),
        "Ayırıcı": (20, 700),
        "Direk": (5, 200),
        "Kablo": (50, 1200),
        "Dağıtım Merkezi": (1000, 9000),
        "Hücre": (100, 2000)
    }

    min_val, max_val = base_ranges.get(network_element_type, (20, 1000))
    affected = random.randint(min_val, max_val)

    if outage_type == "planned":
        affected = int(affected * np.random.uniform(0.8, 1.4))
    else:
        affected = int(affected * np.random.uniform(0.6, 1.8))

    return max(1, affected)


def calculate_energy_not_supplied(duration_min: int, affected_customer_count: int) -> float:
    """
    Basit ENS simülasyonu.
    Ortalama müşteri güç talebi varsayımıyla yaklaşık kWh hesaplanır.
    """

    avg_kw_per_customer = np.random.uniform(0.4, 1.8)
    ens = affected_customer_count * avg_kw_per_customer * (duration_min / 60)
    return round(ens, 2)


def normalize_tr_chars(text: str) -> str:
    replacements = {
        "İ": "I",
        "ı": "i",
        "Ş": "S",
        "ş": "s",
        "Ğ": "G",
        "ğ": "g",
        "Ü": "U",
        "ü": "u",
        "Ö": "O",
        "ö": "o",
        "Ç": "C",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# =============================================================================
# CUSTOMER REGION VERİSİ
# =============================================================================

def generate_customer_region_data() -> pd.DataFrame:
    rows = []

    for city, districts in CITIES_DISTRICTS.items():
        for district in districts:
            total_customer_count = random.randint(15000, 350000)

            residential_ratio = np.random.uniform(0.65, 0.82)
            commercial_ratio = np.random.uniform(0.12, 0.25)
            industrial_ratio = max(0.03, 1 - residential_ratio - commercial_ratio)

            residential_customer_count = int(total_customer_count * residential_ratio)
            commercial_customer_count = int(total_customer_count * commercial_ratio)
            industrial_customer_count = total_customer_count - residential_customer_count - commercial_customer_count

            rows.append({
                "city": city,
                "district": district,
                "total_customer_count": total_customer_count,
                "residential_customer_count": residential_customer_count,
                "commercial_customer_count": commercial_customer_count,
                "industrial_customer_count": industrial_customer_count
            })

    return pd.DataFrame(rows)


# =============================================================================
# WEATHER VERİSİ
# =============================================================================

def generate_weather_data() -> pd.DataFrame:
    rows = []

    current_date = START_DATE.date()
    end_date = END_DATE.date()

    while current_date <= end_date:
        month = current_date.month

        for city, districts in CITIES_DISTRICTS.items():
            for district in districts:

                if month in [12, 1, 2]:
                    temperature = np.random.normal(loc=2, scale=7)
                    precipitation = max(0, np.random.exponential(scale=4))
                    wind_speed = max(0, np.random.normal(loc=18, scale=8))
                elif month in [6, 7, 8]:
                    temperature = np.random.normal(loc=28, scale=5)
                    precipitation = max(0, np.random.exponential(scale=1.5))
                    wind_speed = max(0, np.random.normal(loc=12, scale=5))
                else:
                    temperature = np.random.normal(loc=15, scale=6)
                    precipitation = max(0, np.random.exponential(scale=2.5))
                    wind_speed = max(0, np.random.normal(loc=14, scale=6))

                storm_flag = 1 if wind_speed > 35 or precipitation > 18 else 0

                rows.append({
                    "date": current_date,
                    "city": city,
                    "district": district,
                    "temperature": round(float(temperature), 2),
                    "wind_speed": round(float(wind_speed), 2),
                    "precipitation_mm": round(float(precipitation), 2),
                    "storm_flag": storm_flag
                })

        current_date += timedelta(days=1)

    return pd.DataFrame(rows)


# =============================================================================
# OUTAGE VERİSİ
# =============================================================================

def generate_outage_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    weather_lookup = weather_df.set_index(["date", "city", "district"]).to_dict("index")

    for outage_id in range(1, OUTAGE_COUNT + 1):
        city = random.choice(list(CITIES_DISTRICTS.keys()))
        district = random.choice(CITIES_DISTRICTS[city])

        started_at = random_date(START_DATE, END_DATE)
        outage_date = started_at.date()

        weather_info = weather_lookup.get((outage_date, city, district), {})

        storm_flag = weather_info.get("storm_flag", 0)
        wind_speed = weather_info.get("wind_speed", 0)
        precipitation_mm = weather_info.get("precipitation_mm", 0)

        if storm_flag == 1:
            outage_type = np.random.choice(OUTAGE_TYPES, p=[0.18, 0.82])
            cause = np.random.choice(
                CAUSES,
                p=[0.33, 0.15, 0.05, 0.14, 0.10, 0.06, 0.06, 0.04, 0.05, 0.02]
            )
        else:
            outage_type = np.random.choice(OUTAGE_TYPES, p=[0.38, 0.62])
            cause = np.random.choice(
                CAUSES,
                p=[0.08, 0.18, 0.22, 0.13, 0.10, 0.08, 0.05, 0.04, 0.04, 0.08]
            )

        network_element_type = random.choice(NETWORK_ELEMENT_TYPES)
        source = random.choice(SOURCES)

        duration_min = calculate_duration_minutes(outage_type, cause)
        ended_at = started_at + timedelta(minutes=duration_min)

        affected_customer_count = calculate_affected_customers(network_element_type, outage_type)
        energy_not_supplied_kwh = calculate_energy_not_supplied(duration_min, affected_customer_count)

        is_planned = 1 if outage_type == "planned" else 0

        is_force_majeure = 1 if (
            cause in FORCE_MAJEURE_CAUSES
            and (
                storm_flag == 1
                or wind_speed > 35
                or precipitation_mm > 20
            )
        ) else 0

        feeder = generate_feeder(city, district)
        transformer = generate_transformer(city, district)

        high_impact = 1 if (
            affected_customer_count >= 1000
            or duration_min >= 180
            or energy_not_supplied_kwh >= 1500
        ) else 0

        rows.append({
            "outage_id": outage_id,
            "city": city,
            "district": district,
            "feeder": feeder,
            "transformer": transformer,
            "network_element_type": network_element_type,
            "outage_type": outage_type,
            "source": source,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_min": duration_min,
            "affected_customer_count": affected_customer_count,
            "energy_not_supplied_kwh": energy_not_supplied_kwh,
            "cause": cause,
            "is_planned": is_planned,
            "is_force_majeure": is_force_majeure,
            "storm_flag": storm_flag,
            "wind_speed": round(float(wind_speed), 2),
            "precipitation_mm": round(float(precipitation_mm), 2),
            "high_impact": high_impact
        })

    return pd.DataFrame(rows)


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    print("Veri üretimi başladı...")

    customer_region_df = generate_customer_region_data()
    weather_df = generate_weather_data()
    outages_df = generate_outage_data(weather_df)

    customer_region_path = os.path.join(DATA_DIR, "customer_region.csv")
    weather_path = os.path.join(DATA_DIR, "weather.csv")
    outages_path = os.path.join(DATA_DIR, "outages.csv")

    customer_region_df.to_csv(customer_region_path, index=False, encoding="utf-8-sig")
    weather_df.to_csv(weather_path, index=False, encoding="utf-8-sig")
    outages_df.to_csv(outages_path, index=False, encoding="utf-8-sig")

    print("Veri üretimi tamamlandı.")
    print(f"Customer region dosyası: {customer_region_path}")
    print(f"Weather dosyası: {weather_path}")
    print(f"Outages dosyası: {outages_path}")

    print("\nOutage veri özeti:")
    print(outages_df.head())
    print("\nSatır sayıları:")
    print(f"outages.csv: {len(outages_df)}")
    print(f"weather.csv: {len(weather_df)}")
    print(f"customer_region.csv: {len(customer_region_df)}")


if __name__ == "__main__":
    main()