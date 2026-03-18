# 1 dalis. importuojami naudojami paketai, nustatomi pagrindiniai parametrai

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

akcijos = [
    "ASML.AS", "SAP.DE", "SIE.DE", "MC.PA", "OR.PA",
    "AIR.PA", "SAN.MC", "IBE.MC", "NOVO-B.CO", "SHEL.L"
]

investavimo_suma = 10000
prekybos_dienu_skaicius = 252
svoriu_apribojimai = (0, 0.25)
nerizikingu_palukanu_norma = 0.02  # palukanu norma be rizikos
duomenu_laikotarpio_pradzia = "2021-01-01"
testavimo_pradzia = "2024-01-01"
testavimo_pabaiga = "2026-01-01"
lango_ilgis_men = 36
sandorio_kaina_proc = 0.001
grazos_mazinimas = 0.3

# 2 dalis. duomenu atsisiuntimas

def parsisiusti_duomenis(akcijos, pradzia, pabaiga):
    kainos = yf.download(
        akcijos,
        start=pradzia,
        end=pabaiga,
        progress=False,
        auto_adjust=True
    )["Close"]

    if isinstance(kainos, pd.Series):
        pavadinimas = akcijos if isinstance(akcijos, str) else akcijos[0]
        kainos = kainos.to_frame(name=pavadinimas)

    return kainos.dropna(how="all")

# 3 dalis. skaiciuojamos grazos ir nustatomos dato portfelio perbalansavimui 

def skaiciuoti_grazas(kainos):
    return kainos.pct_change().dropna()

def menesio_pradzios_datos(kainos):
    datos = pd.Series(kainos.index, index=kainos.index)
    return datos.groupby(datos.dt.to_period("M")).first().tolist()

# 4 dalis. apskaiciuojama portfelio apyvarta. 

def apyvarta(seni_svoriai, nauji_svoriai):
    if seni_svoriai is None:
        return np.abs(nauji_svoriai).sum()
    return np.abs(nauji_svoriai - seni_svoriai).sum()
