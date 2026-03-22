# 1. Paketai ir pagrindiniai nustatymai
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
svoriu_apribojimai = (0.0, 0.25)
nerizikingu_palukanu_norma = 0.02
duomenu_laikotarpio_pradzia = "2021-01-01"
testavimo_pradzia = "2024-01-01"
testavimo_pabaiga = "2026-01-01"
lango_ilgis_men = 36
sandorio_kaina_proc = 0.001
grazos_mazinimas = 0.30
min_train_dienu = 60


# 2. Duomenų parsisiuntimas

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

    kainos = kainos.dropna(how="all").sort_index()
    return kainos



# 3. Pagalbinės funkcijos

def skaiciuoti_grazas(kainos):
    return kainos.pct_change().dropna(how="any")

def menesio_pradzios_datos(kainos):
    datos = pd.Series(kainos.index, index=kainos.index)
    return datos.groupby(datos.dt.to_period("M")).first().tolist()



# 4. Portfelio apyvarta
def apyvarta(seni_svoriai, nauji_svoriai, skaiciuoti_pradine_apyvarta=True):
    if seni_svoriai is None:
        return 0.5 * float(np.abs(nauji_svoriai).sum()) if skaiciuoti_pradine_apyvarta else 0.0

    bendras_indeksas = seni_svoriai.index.union(nauji_svoriai.index)
    seni = seni_svoriai.reindex(bendras_indeksas, fill_value=0.0)
    nauji = nauji_svoriai.reindex(bendras_indeksas, fill_value=0.0)

    return 0.5 * float(np.abs(nauji - seni).sum())

# 5. Maksimalaus Sharpe optimizavimas
def optimizuoti_max_sharpe_svorius(grazos):
    aktyvai = list(grazos.columns)
    aktyvu_kiekis = len(aktyvai)

    max_svoris = svoriu_apribojimai[1]
    min_reikalingu_aktyvu = int(np.ceil(1 / max_svoris))
    if aktyvu_kiekis < min_reikalingu_aktyvu:
        return None

    tiketinos_grazos = grazos.mean() * grazos_mazinimas
    kovariaciju_matrica = grazos.cov()

    if tiketinos_grazos.isna().any() or kovariaciju_matrica.isna().any().any():
        return None

    rf_dieninis = (1 + nerizikingu_palukanu_norma) ** (1 / prekybos_dienu_skaicius) - 1

    def tikslas(w):
        w = np.asarray(w)
        portfelio_graza = float(np.dot(w, tiketinos_grazos.values))
        portfelio_var = float(w @ kovariaciju_matrica.values @ w)

        if portfelio_var <= 0:
            return 1e6

        portfelio_vol = np.sqrt(portfelio_var)
        sharpe = (portfelio_graza - rf_dieninis) / portfelio_vol
        return -sharpe

    x0 = np.ones(aktyvu_kiekis) / aktyvu_kiekis

    rezultatas = minimize(
        tikslas,
        x0=x0,
        method="SLSQP",
        bounds=[svoriu_apribojimai] * aktyvu_kiekis,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"maxiter": 500, "ftol": 1e-12}
    )

    if not rezultatas.success:
        print(f"Optimizacija nepavyko: {rezultatas.message}")
        return None

    svoriai = pd.Series(rezultatas.x, index=aktyvai)
    svoriai[svoriai < 1e-10] = 0.0

    suma = svoriai.sum()
    if suma <= 0:
        return None

    svoriai = svoriai / suma

    if (
        abs(svoriai.sum() - 1) > 1e-6
        or (svoriai < svoriu_apribojimai[0] - 1e-8).any()
        or (svoriai > svoriu_apribojimai[1] + 1e-8).any()
    ):
        return None

    return svoriai



# 6. Investuojamų aktyvų atranka

def atrinkti_aktyvus(train, test, min_train_dienu=60):
    aktyvai = []
    for c in train.columns:
        if train[c].notna().sum() >= min_train_dienu and test[c].notna().sum() > 0:
            aktyvai.append(c)
    return aktyvai



# 7. testavimas
def testuoti_strategija(kainos):
    testavimo_kainos = kainos[
        (kainos.index >= pd.Timestamp(testavimo_pradzia)) &
        (kainos.index < pd.Timestamp(testavimo_pabaiga))
    ]

    datos = menesio_pradzios_datos(testavimo_kainos)

    kapitalas = investavimo_suma
    kiekiai = None  # pd.Series su ticker indeksu

    portfelio_verte_laike = []
    svoriu_istorija = []
    apyvartos_istorija = []
    islaidu_istorija = []

    max_svoris = svoriu_apribojimai[1]
    min_reikalingu_aktyvu = int(np.ceil(1 / max_svoris))

    for data in datos:
        men_pr = pd.Timestamp(data)
        men_pb = men_pr + pd.offsets.MonthEnd(1)
        lango_pr = men_pr - pd.DateOffset(months=lango_ilgis_men)

        train_kainos = kainos[
            (kainos.index >= lango_pr) &
            (kainos.index < men_pr)
        ]

        test_kainos = kainos[
            (kainos.index >= men_pr) &
            (kainos.index <= men_pb)
        ]

        aktyvai = atrinkti_aktyvus(train_kainos, test_kainos, min_train_dienu=min_train_dienu)

        if len(aktyvai) < min_reikalingu_aktyvu:
            continue

        train = train_kainos[aktyvai]
        test = test_kainos[aktyvai]

        # Kadangi biržų kalendoriai skiriasi, trūkstamas reikšmes švelniai užpildome į priekį.
        # Po to išmetame tik tas eilutes, kur vis tiek liko trūkumų.
        train = train.ffill().dropna(how="any")
        test = test.ffill().dropna(how="any")

        if len(train) < min_train_dienu or len(test) == 0:
            continue

        train_grazos = skaiciuoti_grazas(train)
        if len(train_grazos) < 20:
            continue

        tiksliniai_svoriai = optimizuoti_max_sharpe_svorius(train_grazos)
        if tiksliniai_svoriai is None:
            continue

        pradzios_kainos = test.iloc[0]

        if kiekiai is None:
            verte = float(kapitalas)
            dabartiniai_svoriai = None
        else:
            turimos_kainos = pradzios_kainos.reindex(kiekiai.index).dropna()
            if turimos_kainos.empty:
                continue

            poziciju_vertes = kiekiai.reindex(turimos_kainos.index, fill_value=0.0) * turimos_kainos
            verte = float(poziciju_vertes.sum())

            if verte > 0:
                dabartiniai_svoriai = (
                    poziciju_vertes / verte
                ).reindex(tiksliniai_svoriai.index, fill_value=0.0)
            else:
                dabartiniai_svoriai = pd.Series(0.0, index=tiksliniai_svoriai.index)

        apyvartos_reiksme = apyvarta(dabartiniai_svoriai, tiksliniai_svoriai)
        islaidos = verte * apyvartos_reiksme * sandorio_kaina_proc

        investuojama = verte - islaidos
        if investuojama <= 0:
            continue

        kiekiai = (investuojama * tiksliniai_svoriai) / pradzios_kainos
        kiekiai = kiekiai.fillna(0.0)

        portfelis = test.mul(kiekiai.reindex(test.columns, fill_value=0.0), axis=1).sum(axis=1)
        if len(portfelis) == 0:
            continue

        kapitalas = float(portfelis.iloc[-1])

        portfelio_verte_laike.append(portfelis)
        svoriu_istorija.append(tiksliniai_svoriai.rename(men_pr))
        apyvartos_istorija.append({"Data": men_pr, "Apyvarta": apyvartos_reiksme})
        islaidu_istorija.append({"Data": men_pr, "Islaidos": islaidos})

    if not portfelio_verte_laike:
        raise ValueError("Nepakanka duomenų backtest'ui.")

    verte = pd.concat(portfelio_verte_laike).sort_index()
    verte = verte[~verte.index.duplicated(keep="last")]

    svoriai_df = pd.DataFrame(svoriu_istorija).fillna(0.0)
    apyvartos_df = pd.DataFrame(apyvartos_istorija)
    islaidos_df = pd.DataFrame(islaidu_istorija)

    return verte, svoriai_df, apyvartos_df, islaidos_df

# 8. Rezultatų įvertinimas
def ivertinti_rezultatus(verte, svoriai_df, apyvartos_df, islaidos_df):
    dienos_grazos = verte.pct_change().dropna()

    bendra_graza = verte.iloc[-1] / investavimo_suma - 1

    metai = (verte.index[-1] - verte.index[0]).days / 365.25
    metine_graza = (
        (verte.iloc[-1] / investavimo_suma) ** (1 / metai) - 1
        if metai > 0 else np.nan
    )

    metinis_kintamumas = (
        dienos_grazos.std() * np.sqrt(prekybos_dienu_skaicius)
        if len(dienos_grazos) > 1 else np.nan
    )

    didziausias_nuosmukis = (verte / verte.cummax() - 1).min()

    efektyvus_aktyvu_skaicius = (
        (1 / svoriai_df.pow(2).sum(axis=1)).mean()
        if not svoriai_df.empty else np.nan
    )

    rf_dieninis = (1 + nerizikingu_palukanu_norma) ** (1 / prekybos_dienu_skaicius) - 1
    if len(dienos_grazos) > 1 and dienos_grazos.std() > 0:
        pertekline_dienos_graza = dienos_grazos - rf_dieninis
        sharpe = np.sqrt(prekybos_dienu_skaicius) * pertekline_dienos_graza.mean() / pertekline_dienos_graza.std()
    else:
        sharpe = np.nan

    calmar = (
        metine_graza / abs(didziausias_nuosmukis)
        if pd.notna(didziausias_nuosmukis) and didziausias_nuosmukis != 0 else np.nan
    )

    return {
        "Galutinė portfelio vertė": float(verte.iloc[-1]),
        "Bendra grąža": float(bendra_graza),
        "Metinė grąža": float(metine_graza) if pd.notna(metine_graza) else np.nan,
        "Metinis kintamumas": float(metinis_kintamumas) if pd.notna(metinis_kintamumas) else np.nan,
        "Šarpo rodiklis": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Didžiausias nuosmukis": float(didziausias_nuosmukis),
        "Kalmaro rodiklis": float(calmar) if pd.notna(calmar) else np.nan,
        "Vidutinė apyvarta": float(apyvartos_df["Apyvarta"].mean()) if not apyvartos_df.empty else np.nan,
        "Visos perbalansavimo išlaidos (€)": float(islaidu_istorija["Islaidos"].sum()) if False else float(islaidos_df["Islaidos"].sum()) if not islaidos_df.empty else np.nan,
        "Efektyvus aktyvų skaičius": float(efektyvus_aktyvu_skaicius) if pd.notna(efektyvus_aktyvu_skaicius) else np.nan,
    }



# 9. Rezultatų spausdinimas
def spausdinti_rezultatus(rezultatai, svoriai_df):
    procentiniai = {
        "Bendra grąža", "Metinė grąža", "Metinis kintamumas",
        "Didžiausias nuosmukis", "Vidutinė apyvarta"
    }
    santykiniai = {"Šarpo rodiklis", "Kalmaro rodiklis", "Efektyvus aktyvų skaičius"}

    print("Paskutiniai svoriai:")
    if not svoriai_df.empty:
        print(svoriai_df.tail(1).T.round(4))
    else:
        print("Nėra svorių istorijos.")

    print("\nRezultatai:")
    for raktas, reiksme in rezultatai.items():
        if pd.isna(reiksme):
            print(f"{raktas}: nan")
        elif raktas in procentiniai:
            print(f"{raktas}: {reiksme:.2%}")
        elif raktas in santykiniai:
            print(f"{raktas}: {reiksme:.3f}")
        else:
            print(f"{raktas}: {reiksme:.2f}")



# 10. Paleidimas ir grafikas

kainos = parsisiusti_duomenis(akcijos, duomenu_laikotarpio_pradzia, testavimo_pabaiga)

verte, svoriai_df, apyvartos_df, islaidos_df = testuoti_strategija(kainos)
rezultatai = ivertinti_rezultatus(verte, svoriai_df, apyvartos_df, islaidos_df)

spausdinti_rezultatus(rezultatai, svoriai_df)

plt.figure(figsize=(12, 6))
plt.plot(verte.index, verte, linewidth=2.5, label="Max Sharpe portfelis")
plt.title("Maksimalaus Šarpo portfelio vertės kitimas")
plt.xlabel("Data")
plt.ylabel("Vertė (€)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
