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

# 2 dalis. 
