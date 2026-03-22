# konstruojamas maksimalaus šarpo rodiklio portfelis

# 1 dalis. importuojami naudojami paketai, nustatomi pagrindiniai parametrai, parenkamos akcijos

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

# 3 dalis. skaiciuojamos grazos ir nustatomos datos portfelio perbalansavimui 

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

# 5 dalis. optimizuojami svoriai maksimalizuojant šarpo rodiklį

def optimizuoti_svorius(grazos):
    tiketinos_grazos = grazos.mean().values * grazos_mazinimas
    kovariaciju_matrica = grazos.cov().values
    aktyvu_kiekis = len(grazos.columns)

    def tikslas(w):
        var = w @ kovariaciju_matrica @ w
        if var <= 0:
            return 1e6

        graza = (w @ tiketinos_grazos) * prekybos_dienu_skaicius
        vol = np.sqrt(var * prekybos_dienu_skaicius)
        sharpe = (graza - nerizikingu_palukanu_norma) / vol

        return -sharpe

    rezultatas = minimize(
        tikslas,
        x0=np.ones(aktyvu_kiekis) / aktyvu_kiekis,
        method="SLSQP",
        bounds=[svoriu_apribojimai] * aktyvu_kiekis,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )

    if not rezultatas.success:
        return np.ones(aktyvu_kiekis) / aktyvu_kiekis

    return rezultatas.x

# 6 dalis svoriu optimizavimas

def optimizuoti_svorius(grazos):
    tiketinos_grazos = grazos.mean().values * grazos_mazinimas
    kovariaciju_matrica = grazos.cov().values
    aktyvu_kiekis = len(grazos.columns)

    def tikslas(w):
        var = w @ kovariaciju_matrica @ w
        if var <= 0:
            return 1e6

        graza = (w @ tiketinos_grazos) * prekybos_dienu_skaicius
        vol = np.sqrt(var * prekybos_dienu_skaicius)
        sharpe = (graza - nerizikingu_palukanu_norma) / vol

        return -sharpe

    rezultatas = minimize(
        tikslas,
        x0=np.ones(aktyvu_kiekis) / aktyvu_kiekis,
        method="SLSQP",
        bounds=[svoriu_apribojimai] * aktyvu_kiekis,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )

    if not rezultatas.success:
        return np.ones(aktyvu_kiekis) / aktyvu_kiekis

    return rezultatas.x

#7 dalis. 

def testuoti_strategija(kainos):
    testavimo_kainos = kainos[
        (kainos.index >= pd.Timestamp(testavimo_pradzia)) &
        (kainos.index < pd.Timestamp(testavimo_pabaiga))
    ]

    datos = menesio_pradzios_datos(testavimo_kainos)

    kapitalas = investavimo_suma
    kiekiai = None

    portfelio_verte_laike = []
    svoriu_istorija = []
    apyvartos_istorija = []
    islaidu_istorija = []

    for data in datos:
        men_pr = data
        men_pb = data + pd.offsets.MonthEnd(1)
        lango_pr = data - pd.DateOffset(months=lango_ilgis_men)

        train = kainos[
            (kainos.index >= lango_pr) &
            (kainos.index < men_pr)
        ]

        test = kainos[
            (kainos.index >= men_pr) &
            (kainos.index <= men_pb)
        ]

# 8 dalis
        aktyvai = [
            c for c in kainos.columns
            if train[c].notna().sum() >= 60 and test[c].notna().sum() > 0
        ]

        if len(aktyvai) < 2:
            continue

        train = train[aktyvai].dropna()
        test = test[aktyvai].dropna()

        if len(train) < 60 or len(test) == 0:
            continue

        tiksliniai_svoriai = optimizuoti_svorius(skaiciuoti_grazas(train))
        pradzios_kainos = test.iloc[0].values

        if kiekiai is None:
            verte = kapitalas
            dabartiniai_svoriai = None
        else:
            poziciju_vertes = kiekiai * pradzios_kainos
            verte = poziciju_vertes.sum()
            dabartiniai_svoriai = (
                poziciju_vertes / verte if verte > 0 else np.zeros(len(tiksliniai_svoriai))
            )

        apyvartos_reiksme = apyvarta(dabartiniai_svoriai, tiksliniai_svoriai)
        islaidos = verte * apyvartos_reiksme * sandorio_kaina_proc

        investuojama = verte - islaidos
        kiekiai = investuojama * tiksliniai_svoriai / pradzios_kainos


# 9 dalis
        investuojama = verte - islaidos
        kiekiai = investuojama * tiksliniai_svoriai / pradzios_kainos

        portfelis = test.mul(kiekiai, axis=1).sum(axis=1)
        kapitalas = portfelis.iloc[-1]

        portfelio_verte_laike.append(portfelis)
        svoriu_istorija.append(pd.Series(tiksliniai_svoriai, index=aktyvai, name=men_pr))
        apyvartos_istorija.append({"Data": men_pr, "Apyvarta": apyvartos_reiksme})
        islaidu_istorija.append({"Data": men_pr, "Islaidos": islaidos})

    if not portfelio_verte_laike:
        raise ValueError("Nepakanka duomenų backtest'ui.")

    verte = pd.concat(portfelio_verte_laike)
    verte = verte[~verte.index.duplicated(keep="first")]

    svoriai_df = pd.DataFrame(svoriu_istorija).fillna(0)
    apyvartos_df = pd.DataFrame(apyvartos_istorija)
    islaidos_df = pd.DataFrame(islaidu_istorija)

    return verte, svoriai_df, apyvartos_df, islaidos_df

# 10 dalis

def ivertinti_rezultatus(verte, svoriai_df, apyvartos_df, islaidos_df):
    dienos_grazos = verte.pct_change().dropna()

    bendra_graza = verte.iloc[-1] / verte.iloc[0] - 1

    metai = (verte.index[-1] - verte.index[0]).days / 365.25
    metine_graza = (verte.iloc[-1] / verte.iloc[0]) ** (1 / metai) - 1 if metai > 0 else np.nan

    metinis_kintamumas = dienos_grazos.std() * np.sqrt(prekybos_dienu_skaicius)
    didziausias_nuosmukis = (verte / verte.cummax() - 1).min()

    efektyvus_aktyvu_skaicius = (
        (1 / svoriai_df.pow(2).sum(axis=1)).mean()
        if not svoriai_df.empty else np.nan
    )

    sharpe = (
        (metine_graza - nerizikingu_palukanu_norma) / metinis_kintamumas
        if metinis_kintamumas and metinis_kintamumas > 0 else np.nan
    )

    calmar = (
        metine_graza / abs(didziausias_nuosmukis)
        if didziausias_nuosmukis != 0 else np.nan
    )

    return {
        "Galutinė portfelio vertė": verte.iloc[-1],
        "Bendra grąža": bendra_graza,
        "Metinė grąža": metine_graza,
        "Metinis kintamumas": metinis_kintamumas,
        "Šarpo rodiklis": sharpe,
        "Didžiausias nuosmukis": didziausias_nuosmukis,
        "Kalmaro rodiklis": calmar,
        "Vidutinė apyvarta": apyvartos_df["Apyvarta"].mean() if not apyvartos_df.empty else np.nan,
        "Visos perbalansavimo išlaidos (€)": islaidos_df["Islaidos"].sum() if not islaidos_df.empty else np.nan,
        "Efektyvus aktyvų skaičius": efektyvus_aktyvu_skaicius,
    }


def spausdinti_rezultatus(rezultatai, svoriai_df):
    procentiniai = {
        "Bendra grąža", "Metinė grąža", "Metinis kintamumas",
        "Didžiausias nuosmukis", "Vidutinė apyvarta"
    }
    santykiniai = {"Šarpo rodiklis", "Kalmaro rodiklis", "Efektyvus aktyvų skaičius"}

    print("Paskutiniai svoriai:")
    print(svoriai_df.tail(1).T.round(4))

    print("\nRezultatai:")
    for raktas, reiksme in rezultatai.items():
        if raktas in procentiniai:
            print(f"{raktas}: {reiksme:.2%}")
        elif raktas in santykiniai:
            print(f"{raktas}: {reiksme:.3f}")
        else:
            print(f"{raktas}: {reiksme:.2f}")


kainos = parsisiusti_duomenis(akcijos, duomenu_laikotarpio_pradzia, testavimo_pabaiga)

verte, svoriai_df, apyvartos_df, islaidos_df = testuoti_strategija(kainos)
rezultatai = ivertinti_rezultatus(verte, svoriai_df, apyvartos_df, islaidos_df)

spausdinti_rezultatus(rezultatai, svoriai_df)

#vizualizacija

plt.figure(figsize=(12, 6))
plt.plot(verte.index, verte, linewidth=2.5, label="Sharpe portfelis")
plt.title("Portfelio vertės kitimas")
plt.xlabel("Data")
plt.ylabel("Vertė (€)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
