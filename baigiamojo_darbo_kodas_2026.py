# importuojamos reikalingos bibliotekos
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message=r".*Timestamp\.utcnow is deprecated.*") # kad nerodytų klaidos
try:
    rodyti = display
except NameError:
    rodyti = print

# atsisiunciami duomenys ir apskaiciuojamos grazos
akcijos = ["SPY", "IJH", "VEA", "EEM", "BND", "TLT", "TIP", "HYG", "GLD", "VNQ"]
duomenys = yf.download(akcijos, start="2013-01-01", end="2026-01-01", auto_adjust=True, progress=False)
kainos = duomenys["Close"].dropna(how="all").ffill()[akcijos]
graza = kainos.pct_change().dropna()
visata, kainos, grazos = akcijos.copy(), kainos.copy(), graza.copy()

#  nustatyti parametrai
prekybos_dienos = 252
bazinis_laikotarpis = 252
ewma_pusperiodis = 30
metine_nerizikinga_palukanu_norma = 0.0
sandoriu_sanaudu_norma = 0.001

# sutvarkomi duomenys, kad atitiktų aktyvus ir indeksus
visata = [s for s in visata if s in grazos.columns]
grazos = grazos[visata].copy()
kainos = kainos[visata].copy()
grazos.index = pd.to_datetime(grazos.index)
kainos.index = pd.to_datetime(kainos.index)
grazos = grazos.sort_index()
kainos = kainos.sort_index()

if len(visata) < 2:
    raise ValueError("Nepakanka tinkamų aktyvų analizei atlikti.")

# normalizuoja ir sutvarko svorius, kad būtų tinkami optimizacijai
def isvalyti(svoriai):
    svoriai = np.asarray(svoriai, dtype=float)
    svoriai = np.nan_to_num(svoriai, nan=0.0, posinf=0.0, neginf=0.0)
    svoriai = np.clip(svoriai, 0.0, 1.0)
    suma = svoriai.sum()
    if suma <= 1e-12:
        return np.repeat(1 / len(svoriai), len(svoriai))
    return svoriai / suma
# užtikrina, kad kovariacijų matrica būtų tinkama optimizacijai
def saugi_kovariacija(kovariacija):
    kovariacija = np.asarray(kovariacija, dtype=float)
    kovariacija = np.nan_to_num(kovariacija, nan=0.0, posinf=0.0, neginf=0.0)
    kovariacija = 0.5 * (kovariacija + kovariacija.T)
    istrizaine = np.diag(kovariacija)
    if (istrizaine < 0).any():
        kovariacija += np.eye(kovariacija.shape[0]) * (abs(istrizaine.min()) + 1e-10)
    return kovariacija + np.eye(kovariacija.shape[0]) * 1e-10

# Ši funkcija apskaičiuoja vertinimo rodiklius
def rodikliai(grazos_eilute, prekybos_dienu_sk=prekybos_dienos, nerizikinga_norma=metine_nerizikinga_palukanu_norma):
    grazos_eilute = pd.Series(grazos_eilute).dropna()
    if len(grazos_eilute) < 2:
        return np.nan, np.nan, np.nan, np.nan
    metine_graza = (1 + grazos_eilute).prod() ** (prekybos_dienu_sk / len(grazos_eilute)) - 1
    metinis_kintamumas = grazos_eilute.std() * np.sqrt(prekybos_dienu_sk)
    sarpo_rodiklis = (metine_graza - nerizikinga_norma) / metinis_kintamumas if metinis_kintamumas > 0 else np.nan
    sukaupta_graza = (1 + grazos_eilute).cumprod()
    maksimalus_nuosmukis = (sukaupta_graza / sukaupta_graza.cummax() - 1).min()
    return metine_graza, metinis_kintamumas, sarpo_rodiklis, maksimalus_nuosmukis

# Funkcija apskaičiuoja apyvartą.
def apyvarta(ankstesni, nauji, laikotarpio_grazos):
    ankstesni = isvalyti(np.asarray(ankstesni, dtype=float))
    nauji = isvalyti(np.asarray(nauji, dtype=float))
    if laikotarpio_grazos is None or len(laikotarpio_grazos) == 0:
        pasislinke = ankstesni.copy()
    else:
        laikotarpio_grazos = pd.DataFrame(laikotarpio_grazos)
        bendros_grazos = (1 + laikotarpio_grazos).prod().values
        pasislinke = ankstesni * bendros_grazos
        suma = pasislinke.sum()
        pasislinke = pasislinke / suma if suma > 1e-12 else ankstesni.copy()
    return 0.5 * np.abs(nauji - pasislinke).sum()

# Funkcija, kuri sukuria rezultatų lentelę 
def rezultatu_lentele(rodikliu_duomenys, apyvartos_duomenys=None):
    lentele = pd.DataFrame(index=rodikliu_duomenys.keys())
    for pavadinimas, reiksmes in rodikliu_duomenys.items():
        lentele.loc[pavadinimas, "Metinė grąža"] = reiksmes[0]
        lentele.loc[pavadinimas, "Metinis kintamumas"] = reiksmes[1]
        lentele.loc[pavadinimas, "Šarpo rodiklis"] = reiksmes[2]
        lentele.loc[pavadinimas, "Maksimalus nuosmukis"] = reiksmes[3]
        if apyvartos_duomenys is not None:
            lentele.loc[pavadinimas, "Vidutinė mėnesinė apyvarta"] = apyvartos_duomenys.get(pavadinimas, np.nan)
    return lentele

# Funkcija, kuri nustato maksimalias ir minimalias ribas svoriams
def gauti_ribas(kiekis, maksimalus_svoris=None):
    if maksimalus_svoris is None:
        return tuple((0, 1) for _ in range(kiekis))
    if maksimalus_svoris * kiekis < 1:
        raise ValueError("Maksimalus aktyvo svoris yra per mažas.")
    return tuple((0, maksimalus_svoris) for _ in range(kiekis))

# Funkcija, kuri pašalina neleistinas reikšmes
def isvalyti_su_riba(svoriai, maksimalus_svoris=None):
    if maksimalus_svoris is None:
        return isvalyti(svoriai)
    svoriai = np.asarray(svoriai, dtype=float)
    kiekis = len(svoriai)
    if maksimalus_svoris * kiekis < 1:
        raise ValueError("Maksimalus aktyvo svoris yra per mažas.")
    svoriai = np.nan_to_num(svoriai, nan=0.0, posinf=0.0, neginf=0.0)
    svoriai = np.clip(svoriai, 0.0, maksimalus_svoris)
    if svoriai.sum() <= 1e-12:
        return np.repeat(1 / kiekis, kiekis)
    svoriai = svoriai / svoriai.sum()
    for _ in range(100):
        virsija = svoriai > maksimalus_svoris
        if not virsija.any():
            break
        perteklius = (svoriai[virsija] - maksimalus_svoris).sum()
        svoriai[virsija] = maksimalus_svoris
        gali_gauti = ~virsija
        laisva_vieta = maksimalus_svoris - svoriai[gali_gauti]
        laisva_suma = laisva_vieta.sum()
        if laisva_suma <= 1e-12:
            break
        svoriai[gali_gauti] += perteklius * laisva_vieta / laisva_suma
    return svoriai / svoriai.sum()

# Funkcija, kuri apskaičiuoja mažiausios dispersijos portfelio svorius
def maziausia_dispersija(kovariacija, maksimalus_svoris=None, grazinti_sekme=False):
    kovariacija = saugi_kovariacija(kovariacija)
    kiekis = kovariacija.shape[0]
    pradiniai_svoriai = np.repeat(1 / kiekis, kiekis)
    ribos = gauti_ribas(kiekis, maksimalus_svoris)
    apribojimai = {"type": "eq", "fun": lambda svoriai: svoriai.sum() - 1}
    rezultatas = minimize(lambda svoriai: svoriai @ kovariacija @ svoriai, pradiniai_svoriai,
                          method="SLSQP", bounds=ribos, constraints=apribojimai,
                          options={"maxiter": 1000, "ftol": 1e-10})
    if rezultatas.success:
        svoriai = isvalyti_su_riba(rezultatas.x, maksimalus_svoris)
    else:
        svoriai = pradiniai_svoriai
    return (svoriai, bool(rezultatas.success)) if grazinti_sekme else svoriai

# Funkcija, kuri apskaičiuoja didžiausio Šarpo rodiklio portfelio svorius
def didziausias_sarpo_rodiklis(vidurkiai, kovariacija, nerizikinga_norma=metine_nerizikinga_palukanu_norma,
                                prekybos_dienu_sk=prekybos_dienos, maksimalus_svoris=None, grazinti_sekme=False):
    kovariacija = saugi_kovariacija(kovariacija)
    vidurkiai = np.nan_to_num(np.asarray(vidurkiai, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    kiekis = len(vidurkiai)
    dienine_nerizikinga_norma = (1 + nerizikinga_norma) ** (1 / prekybos_dienu_sk) - 1
    pradiniai_svoriai = np.repeat(1 / kiekis, kiekis)
    ribos = gauti_ribas(kiekis, maksimalus_svoris)
    apribojimai = {"type": "eq", "fun": lambda svoriai: svoriai.sum() - 1}
    def tikslo_funkcija(svoriai):
        portfelio_rizika = np.sqrt(max(svoriai @ kovariacija @ svoriai, 0.0))
        if portfelio_rizika <= 1e-12:
            return 1e10
        portfelio_graza = svoriai @ vidurkiai
        return -(portfelio_graza - dienine_nerizikinga_norma) / portfelio_rizika
    rezultatas = minimize(tikslo_funkcija, pradiniai_svoriai, method="SLSQP",
                          bounds=ribos, constraints=apribojimai,
                          options={"maxiter": 1000, "ftol": 1e-10})
    if rezultatas.success:
        svoriai = isvalyti_su_riba(rezultatas.x, maksimalus_svoris)
    else:
        svoriai = pradiniai_svoriai
    return (svoriai, bool(rezultatas.success)) if grazinti_sekme else svoriai

# Funkcija, kuri apskaičiuoja vidurkio-dispersijos portfelio svorius
def vidurkio_dispersijos_naudingumas(vidurkiai, kovariacija, rizikos_vengimas=10.0,
                                     maksimalus_svoris=None, grazinti_sekme=False):
    kovariacija = saugi_kovariacija(kovariacija)
    vidurkiai = np.nan_to_num(np.asarray(vidurkiai, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    kiekis = len(vidurkiai)
    pradiniai_svoriai = np.repeat(1 / kiekis, kiekis)
    ribos = gauti_ribas(kiekis, maksimalus_svoris)
    apribojimai = {"type": "eq", "fun": lambda svoriai: svoriai.sum() - 1}
    def tikslo_funkcija(svoriai):
        return -(svoriai @ vidurkiai - rizikos_vengimas * (svoriai @ kovariacija @ svoriai))
    rezultatas = minimize(tikslo_funkcija, pradiniai_svoriai, method="SLSQP",
                          bounds=ribos, constraints=apribojimai,
                          options={"maxiter": 1000, "ftol": 1e-10})
    if rezultatas.success:
        svoriai = isvalyti_su_riba(rezultatas.x, maksimalus_svoris)
    else:
        svoriai = pradiniai_svoriai
    return (svoriai, bool(rezultatas.success)) if grazinti_sekme else svoriai

# Funkcija, kuri apskaičiuoja eksponentiškai svertą kovariacijos matricą
def ewma_kovariacija(grazos_duomenys, pusperiodis=ewma_pusperiodis):
    grazos_duomenys = pd.DataFrame(grazos_duomenys).dropna(how="all")
    if len(grazos_duomenys) < 2:
        return np.eye(grazos_duomenys.shape[1]) * 1e-10
    kovariacija = grazos_duomenys.ewm(halflife=pusperiodis).cov().loc[grazos_duomenys.index[-1]].values
    return saugi_kovariacija(kovariacija)

# Funkcija, kuri apskaičiuoja Ledoit-Wolf kovariacijos matricą
def ledoit_wolf_kovariacija(grazos_duomenys):
    grazos_duomenys = pd.DataFrame(grazos_duomenys).dropna()
    if len(grazos_duomenys) < 2:
        return np.eye(grazos_duomenys.shape[1]) * 1e-10
    modelis = LedoitWolf().fit(grazos_duomenys.values)
    return saugi_kovariacija(modelis.covariance_)

# Funkcija, kuri apskaičiuoja standartizuotą reikšmę per nurodytą langą
def standartizuota_reiksme(eilute, langas, maziausias_kiekis):
    eilute = pd.Series(eilute).astype(float)
    vidurkis = eilute.rolling(langas, min_periods=maziausias_kiekis).mean()
    nuokrypis = eilute.rolling(langas, min_periods=maziausias_kiekis).std()
    return ((eilute - vidurkis) / nuokrypis).replace([np.inf, -np.inf], np.nan)

# Funkcija, kuri apskaičiuoja minkštą maksimumą su temperatūros parametru
def minkstasis_maksimumas(reiksmes, temperatura):
    if temperatura <= 0:
        raise ValueError("Temperatūra turi būti teigiama.")
    reiksmes = np.clip(np.nan_to_num(np.asarray(reiksmes, dtype=float), nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    eksponentes = np.exp((reiksmes - reiksmes.max()) / temperatura)
    suma = eksponentes.sum()
    return eksponentes / suma if suma > 1e-12 else np.repeat(1 / len(reiksmes), len(reiksmes))

# Rinkos signalai ir indekso sukūrimas
rinkos_grazos = grazos[visata].mean(axis=1)
rinkos_indeksas = (1 + rinkos_grazos).cumprod()
rinkos_indeksas.index = pd.to_datetime(rinkos_indeksas.index)
rinkos_indeksas = rinkos_indeksas.sort_index()
rinkos_grazos.index = pd.to_datetime(rinkos_grazos.index)
rinkos_grazos = rinkos_grazos.sort_index()
bendras_indeksas = grazos.index.intersection(rinkos_grazos.dropna().index)
grazos = grazos.loc[bendras_indeksas].copy()
kainos = kainos.loc[bendras_indeksas].copy()
rinkos_indeksas = rinkos_indeksas.loc[bendras_indeksas].copy()
rinkos_grazos = rinkos_grazos.loc[bendras_indeksas].copy()
spx_graza = rinkos_grazos.copy()
spx_sukaupta_graza = rinkos_indeksas.copy()
spx = rinkos_indeksas.copy()

strategiju_pavadinimai = [
    "Vienodi svoriai", "Mažiausia dispersija", "Didžiausias Šarpo rodiklis",
    "Vidurkio-dispersijos", "Režimo mažiausia dispersija",
    "Režimo didžiausias Šarpo rodiklis", "Režimo vidurkio-dispersijos"
]

# Parametrai atgaliniam testui    
z_langas = 752
z_minimalus_stebejimu_kiekis = 252
bazine_temperatura = 1.0
bazinis_tendencijos_svoris = 1/3
bazinis_momento_svoris = 1/3
bazinis_kintamumo_svoris = 1/3
bazinis_rizikos_vengimas = 10.0
agresyvus_rizikos_vengimas = 5.0
neutralus_rizikos_vengimas = 10.0
gynybinis_rizikos_vengimas = 20.0
bazinis_perbalansavimo_daznis = "ME"

# Funkcija
def vykdyti_atgalini_testa(
    temperatura=bazine_temperatura, tendencijos_svoris=bazinis_tendencijos_svoris,
    momento_svoris=bazinis_momento_svoris, kintamumo_svoris=bazinis_kintamumo_svoris,
    laikotarpis=bazinis_laikotarpis, perbalansavimo_daznis=bazinis_perbalansavimo_daznis,
    sandoriu_sanaudu_norma=sandoriu_sanaudu_norma, metine_nerizikinga_norma=metine_nerizikinga_palukanu_norma,
    minimalus_rezimo_aktyvo_svoris=0.00,
    maksimalus_rezimo_aktyvo_svoris=0.40,
    rizikos_vengimas=bazinis_rizikos_vengimas,
    agresyvus_rizikos_vengimas=agresyvus_rizikos_vengimas,
    neutralus_rizikos_vengimas=neutralus_rizikos_vengimas,
    gynybinis_rizikos_vengimas=gynybinis_rizikos_vengimas,
    grazinti_pilna=False
):
    kiekis_aktyvu = len(visata)
    if minimalus_rezimo_aktyvo_svoris * kiekis_aktyvu > 1:
        raise ValueError("Minimalus aktyvo svoris yra per didelis.")
    if maksimalus_rezimo_aktyvo_svoris * kiekis_aktyvu < 1:
        raise ValueError("Maksimalus aktyvo svoris yra per mažas.")
    if minimalus_rezimo_aktyvo_svoris > maksimalus_rezimo_aktyvo_svoris:
        raise ValueError("Minimalus aktyvo svoris negali būti didesnis už maksimalų.")
    # Funkcija, kuri sutvarko režimo svorius pagal nustatytas ribas ir normalizuoja juos
    def sutvarkyti_rezimo_svorius(svoriai):
        svoriai = np.nan_to_num(np.asarray(svoriai, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        kiekis = len(svoriai)
        if svoriai.sum() <= 1e-12:
            svoriai = np.repeat(1 / kiekis, kiekis)
        else:
            svoriai = svoriai / svoriai.sum()
        min_svoris = minimalus_rezimo_aktyvo_svoris
        max_svoris = maksimalus_rezimo_aktyvo_svoris
        baziniai_svoriai = np.repeat(min_svoris, kiekis)
        likutis = 1 - baziniai_svoriai.sum()
        if likutis <= 1e-12:
            return baziniai_svoriai / baziniai_svoriai.sum()
        papildomi_svoriai = np.maximum(svoriai - min_svoris, 0.0)
        if papildomi_svoriai.sum() <= 1e-12:
            papildomi_svoriai = np.repeat(1 / kiekis, kiekis)
        else:
            papildomi_svoriai = papildomi_svoriai / papildomi_svoriai.sum()
        svoriai = baziniai_svoriai + likutis * papildomi_svoriai
        for _ in range(100):
            virsija = svoriai > max_svoris
            if not virsija.any():
                break
            perteklius = (svoriai[virsija] - max_svoris).sum()
            svoriai[virsija] = max_svoris
            gali_gauti = svoriai < max_svoris - 1e-12
            laisva_vieta = max_svoris - svoriai[gali_gauti]
            laisva_suma = laisva_vieta.sum()
            if laisva_suma <= 1e-12:
                break
            svoriai[gali_gauti] += perteklius * laisva_vieta / laisva_suma
        svoriai = np.clip(svoriai, min_svoris, max_svoris)
        return svoriai / svoriai.sum()

    perbalansavimo_datos = pd.DatetimeIndex(
        grazos.index.to_series().resample(perbalansavimo_daznis).last().dropna().values
    )
    perbalansavimo_indeksai = [grazos.index.get_loc(d) for d in perbalansavimo_datos if d in grazos.index]

    bruto = {p: pd.Series(np.nan, index=grazos.index, dtype=float) for p in strategiju_pavadinimai}
    neto = {p: pd.Series(np.nan, index=grazos.index, dtype=float) for p in strategiju_pavadinimai}
    rezimai = pd.DataFrame(index=perbalansavimo_datos, columns=["Agresyvus", "Gynybinis", "Neutralus"], dtype=float)
    apyvartos = pd.DataFrame(index=perbalansavimo_datos, columns=strategiju_pavadinimai, dtype=float)
    svoriu_istorija = {p: pd.DataFrame(index=perbalansavimo_datos, columns=visata, dtype=float) for p in strategiju_pavadinimai}
    klaidos = {"maziausios_dispersijos_klaidos": 0, "didziausio_sarpo_rodiklio_klaidos": 0, "vidurkio_dispersijos_naudingumo_klaidos": 0}
    ankstesni_svoriai = {p: None for p in strategiju_pavadinimai}
    ankstesnis_indeksas = None
    ankstesni_rezimo_svoriai = None

    for numeris, dabartinis_indeksas in enumerate(perbalansavimo_indeksai):
        if dabartinis_indeksas - laikotarpis < 0:
            continue
        data = grazos.index[dabartinis_indeksas]
        laikotarpio_grazos = grazos[visata].iloc[dabartinis_indeksas - laikotarpis:dabartinis_indeksas]
        if len(laikotarpio_grazos) < laikotarpis or laikotarpio_grazos.isna().any().any():
            continue

        # kovariacijos matricų apskaičiavimas
        agresyvi_kovariacija = ewma_kovariacija(laikotarpio_grazos, pusperiodis=ewma_pusperiodis)
        neutrali_kovariacija = saugi_kovariacija(laikotarpio_grazos.cov().values)
        gynybine_kovariacija = ledoit_wolf_kovariacija(laikotarpio_grazos)

        # paprastos tikėtinos grąžos
        vidurkiai = laikotarpio_grazos.mean().values
        istorija_iki_datos = grazos[visata].iloc[:dabartinis_indeksas]
        
        #  ši funkcija grąžina paskutinių nurodyto laikotarpio dienų vidutinę grąžą
        def saugus_vidurkis(duom, langas):           
            if len(duom) >= langas:
                return duom.iloc[-langas:].mean().values
            elif len(duom) > 0:
                return duom.mean().values
            else:
                return np.zeros(duom.shape[1])

        # Apskaičiuojamos tikėtinos grąžos
        mu_agres = saugus_vidurkis(istorija_iki_datos, 126)        
        mean_252 = saugus_vidurkis(istorija_iki_datos, 252)
        mean_756 = saugus_vidurkis(istorija_iki_datos, 756)
        mu_neut = 0.5 * mean_252 + 0.5 * mean_756      
        mu_def = 0.25 * mean_252
        kiekis = len(visata)
        vienodi_svoriai = np.repeat(1 / kiekis, kiekis)

        # statinių portfelių svoriai
        maziausios_dispersijos_svoriai, sekme = maziausia_dispersija(neutrali_kovariacija, grazinti_sekme=True)
        klaidos["maziausios_dispersijos_klaidos"] += int(not sekme)
        didziausio_sarpo_svoriai, sekme = didziausias_sarpo_rodiklis(vidurkiai, neutrali_kovariacija, nerizikinga_norma=metine_nerizikinga_norma, grazinti_sekme=True)
        klaidos["didziausio_sarpo_rodiklio_klaidos"] += int(not sekme)
        vidurkio_dispersijos_svoriai, sekme = vidurkio_dispersijos_naudingumas(vidurkiai, neutrali_kovariacija, rizikos_vengimas=rizikos_vengimas, grazinti_sekme=True)
        klaidos["vidurkio_dispersijos_naudingumo_klaidos"] += int(not sekme)

        # Apskaičiuojami režimo‑specifiniai svoriai
        maziausios_dispersijos_komponentai = [
            sutvarkyti_rezimo_svorius(maziausia_dispersija(agresyvi_kovariacija, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(maziausia_dispersija(gynybine_kovariacija, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(maziausia_dispersija(neutrali_kovariacija, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris))
        ]
        didziausio_sarpo_komponentai = [
            sutvarkyti_rezimo_svorius(didziausias_sarpo_rodiklis(mu_agres, agresyvi_kovariacija, nerizikinga_norma=metine_nerizikinga_norma, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(didziausias_sarpo_rodiklis(mu_def, gynybine_kovariacija, nerizikinga_norma=metine_nerizikinga_norma, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(didziausias_sarpo_rodiklis(mu_neut, neutrali_kovariacija, nerizikinga_norma=metine_nerizikinga_norma, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris))
        ]
        vidurkio_dispersijos_komponentai = [
            sutvarkyti_rezimo_svorius(vidurkio_dispersijos_naudingumas(mu_agres, agresyvi_kovariacija, rizikos_vengimas=agresyvus_rizikos_vengimas, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(vidurkio_dispersijos_naudingumas(mu_def, gynybine_kovariacija, rizikos_vengimas=gynybinis_rizikos_vengimas, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris)),
            sutvarkyti_rezimo_svorius(vidurkio_dispersijos_naudingumas(mu_neut, neutrali_kovariacija, rizikos_vengimas=neutralus_rizikos_vengimas, maksimalus_svoris=maksimalus_rezimo_aktyvo_svoris))
        ]

        # Apskaičiuojami režimo signalai ir svoriai
        signalo_data = grazos.index[dabartinis_indeksas - 1]
        spx_sukaupta_iki_datos = spx_sukaupta_graza.loc[:signalo_data]
        if len(spx_sukaupta_iki_datos) < 252:
            rezimo_svoriai = np.array([0.0, 0.0, 1.0])
        else:
            tendencijos_zaliava = spx_sukaupta_iki_datos.rolling(50).mean() / spx_sukaupta_iki_datos.rolling(200).mean() - 1
            momento_zaliava = spx_sukaupta_iki_datos / spx_sukaupta_iki_datos.shift(252) - 1
            spx_grazos_iki_datos = spx_graza.loc[:signalo_data]
            kintamumas_20 = spx_grazos_iki_datos.rolling(20).std() * np.sqrt(prekybos_dienos)
            kintamumas_252 = spx_grazos_iki_datos.rolling(252).std() * np.sqrt(prekybos_dienos)
            kintamumo_zaliava = kintamumas_20 / kintamumas_252 - 1
            naujausia = pd.Series({
                "tendencija": standartizuota_reiksme(tendencijos_zaliava, z_langas, z_minimalus_stebejimu_kiekis).iloc[-1],
                "momentas": standartizuota_reiksme(momento_zaliava, z_langas, z_minimalus_stebejimu_kiekis).iloc[-1],
                "kintamumas": standartizuota_reiksme(kintamumo_zaliava, z_langas, z_minimalus_stebejimu_kiekis).iloc[-1]
            })
            if naujausia.isna().any():
                rezimo_svoriai = np.array([0.0, 0.0, 1.0])
            else:
                rizikos_patrauklumas = (tendencijos_svoris * naujausia["tendencija"] +
                                        momento_svoris * naujausia["momentas"] -
                                        kintamumo_svoris * naujausia["kintamumas"])
                rizikos_patrauklumas = np.clip(rizikos_patrauklumas, -2.0, 2.0)
                rezimo_svoriai = minkstasis_maksimumas([rizikos_patrauklumas, -rizikos_patrauklumas, -abs(rizikos_patrauklumas)], temperatura)

        
        rezimo_svoriai = isvalyti(rezimo_svoriai)
        rezimai.loc[data] = rezimo_svoriai

        # sujungti režimo svorius su komponentų svoriais
        rezimo_maziausios_dispersijos_svoriai = sutvarkyti_rezimo_svorius(rezimo_svoriai @ np.vstack(maziausios_dispersijos_komponentai))
        rezimo_didziausio_sarpo_svoriai = sutvarkyti_rezimo_svorius(rezimo_svoriai @ np.vstack(didziausio_sarpo_komponentai))
        rezimo_vidurkio_dispersijos_svoriai = sutvarkyti_rezimo_svorius(rezimo_svoriai @ np.vstack(vidurkio_dispersijos_komponentai))

        

        tiksliniai_svoriai = {
            "Vienodi svoriai": vienodi_svoriai,
            "Mažiausia dispersija": maziausios_dispersijos_svoriai,
            "Didžiausias Šarpo rodiklis": didziausio_sarpo_svoriai,
            "Vidurkio-dispersijos": vidurkio_dispersijos_svoriai,
            "Režimo mažiausia dispersija": rezimo_maziausios_dispersijos_svoriai,
            "Režimo didžiausias Šarpo rodiklis": rezimo_didziausio_sarpo_svoriai,
            "Režimo vidurkio-dispersijos": rezimo_vidurkio_dispersijos_svoriai
        }

        dabartine_apyvarta = {}
        for pavadinimas in strategiju_pavadinimai:
            svoriu_istorija[pavadinimas].loc[data] = tiksliniai_svoriai[pavadinimas]
            if ankstesni_svoriai[pavadinimas] is None or ankstesnis_indeksas is None:
                apyvartos.loc[data, pavadinimas] = np.nan
                dabartine_apyvarta[pavadinimas] = 0.0
            else:
                verte = apyvarta(ankstesni_svoriai[pavadinimas], tiksliniai_svoriai[pavadinimas],
                                 grazos[visata].iloc[ankstesnis_indeksas + 1:dabartinis_indeksas + 1])
                apyvartos.loc[data, pavadinimas] = verte
                dabartine_apyvarta[pavadinimas] = verte
        ankstesni_svoriai = {p: tiksliniai_svoriai[p].copy() for p in strategiju_pavadinimai}
        ankstesnis_indeksas = dabartinis_indeksas

        if numeris < len(perbalansavimo_indeksai) - 1:
            kitas_indeksas = perbalansavimo_indeksai[numeris + 1]
        else:
            kitas_indeksas = len(grazos) - 1
        laikymo_grazos = grazos[visata].iloc[dabartinis_indeksas + 1:kitas_indeksas + 1]
        laikymo_datos = grazos.index[dabartinis_indeksas + 1:kitas_indeksas + 1]
        if len(laikymo_grazos) == 0:
            continue

        for pavadinimas in strategiju_pavadinimai:
            bruto_graza = (laikymo_grazos * tiksliniai_svoriai[pavadinimas]).sum(axis=1)
            neto_graza = bruto_graza.copy()
            neto_graza.loc[laikymo_datos[0]] -= dabartine_apyvarta[pavadinimas] * sandoriu_sanaudu_norma
            bruto[pavadinimas].loc[laikymo_datos] = bruto_graza
            neto[pavadinimas].loc[laikymo_datos] = neto_graza

    bruto_rodikliai = {p: rodikliai(bruto[p], nerizikinga_norma=metine_nerizikinga_norma) for p in strategiju_pavadinimai}
    neto_rodikliai = {p: rodikliai(neto[p], nerizikinga_norma=metine_nerizikinga_norma) for p in strategiju_pavadinimai}
    vidutine_apyvarta = apyvartos.mean(skipna=True).to_dict()
    neto_sarpo_rodikliai = {p: neto_rodikliai[p][2] for p in strategiju_pavadinimai}

    if not grazinti_pilna:
        return neto_sarpo_rodikliai

    return {
        "dienines_bruto_grazos": bruto,
        "dienines_neto_grazos": neto,
        "rezimo_komponentu_svoriai": rezimai.dropna(how="all"),
        "apyvarta": apyvartos.dropna(how="all"),
        "vidutine_apyvarta": vidutine_apyvarta,
        "svoriu_istorija": svoriu_istorija,
        "konvergavimo_zurnalas": klaidos,
        "bruto_rodikliai": bruto_rodikliai,
        "neto_rodikliai": neto_rodikliai,
        "neto_sarpo_rodikliai": neto_sarpo_rodikliai
    }

# Vykdomas testavimas ir rodomi rezultatai
pagrindiniai_rezultatai = vykdyti_atgalini_testa(grazinti_pilna=True)
vertinimo_pradzia = pd.Timestamp("2016-01-01")
dienines_bruto_grazos = {p: serija.loc[serija.index >= vertinimo_pradzia] for p, serija in pagrindiniai_rezultatai["dienines_bruto_grazos"].items()}
dienines_neto_grazos = {p: serija.loc[serija.index >= vertinimo_pradzia] for p, serija in pagrindiniai_rezultatai["dienines_neto_grazos"].items()}
rezimo_komponentu_svoriai = pagrindiniai_rezultatai["rezimo_komponentu_svoriai"].loc[pagrindiniai_rezultatai["rezimo_komponentu_svoriai"].index >= vertinimo_pradzia].copy()
apyvartos_lentele = pagrindiniai_rezultatai["apyvarta"].loc[pagrindiniai_rezultatai["apyvarta"].index >= vertinimo_pradzia].copy()
svoriu_istorija = {p: lentele.loc[lentele.index >= vertinimo_pradzia].copy() for p, lentele in pagrindiniai_rezultatai["svoriu_istorija"].items()}
konvergavimo_zurnalas = pagrindiniai_rezultatai["konvergavimo_zurnalas"]
bruto_rodikliai = {p: rodikliai(dienines_bruto_grazos[p], nerizikinga_norma=metine_nerizikinga_palukanu_norma) for p in strategiju_pavadinimai}
neto_rodikliai = {p: rodikliai(dienines_neto_grazos[p], nerizikinga_norma=metine_nerizikinga_palukanu_norma) for p in strategiju_pavadinimai}
vidutine_apyvarta = apyvartos_lentele.mean(skipna=True).to_dict()
bruto_rezultatu_lentele = rezultatu_lentele(bruto_rodikliai, vidutine_apyvarta)
neto_rezultatu_lentele = rezultatu_lentele(neto_rodikliai, vidutine_apyvarta)
print("\nPORTFELIŲ REZULTATAI PRIEŠ SANDORIŲ SĄNAUDAS")
print(f"Vertinimo laikotarpis: nuo {vertinimo_pradzia.date()}")
rodyti(bruto_rezultatu_lentele.round(4))
print("\nPORTFELIŲ REZULTATAI PO SANDORIŲ SĄNAUDŲ")
print(f"Vertinimo laikotarpis: nuo {vertinimo_pradzia.date()}")
rodyti(neto_rezultatu_lentele.round(4))
print("\nOPTIMIZAVIMO KONVERGAVIMO PROBLEMOS")
print(f"Mažiausios dispersijos optimizavimas: {konvergavimo_zurnalas.get('maziausios_dispersijos_klaidos', 0)}")
print(f"Didžiausio Šarpo rodiklio optimizavimas: {konvergavimo_zurnalas.get('didziausio_sarpo_rodiklio_klaidos', 0)}")
print(f"Vidurkio-dispersijos naudingumo optimizavimas: {konvergavimo_zurnalas.get('vidurkio_dispersijos_naudingumo_klaidos', 0)}")
print("\nNAUDOTOS PAGRINDINĖS PRIELAIDOS")
print(f"Vertinimo laikotarpio pradžia: {vertinimo_pradzia.date()}")
print(f"Metinė nerizikinga palūkanų norma: {metine_nerizikinga_palukanu_norma:.2%}")
print(f"Sandorių sąnaudų norma: {sandoriu_sanaudu_norma:.2%}")
print(f"Bazinis vertinimo langas: {bazinis_laikotarpis} prekybos dienos")
print("Rinkos signalams naudojamas indeksas: vienodų svorių pasirinktos akcijų visatos indeksas")
print(f"Agresyvus komponentas: EWMA kovariacijų įvertis, {ewma_pusperiodis} prekybos dienų pusėjimo trukmė; vidurkio-dispersijos naudingumo rizikos vengimas = {agresyvus_rizikos_vengimas}")
print(f"Neutralus komponentas: paprasta imties kovariacijų matrica, 252 prekybos dienos; vidurkio-dispersijos naudingumo rizikos vengimas = {neutralus_rizikos_vengimas}")
print(f"Gynybinis komponentas: Ledoit–Wolf kovariacijų įvertis, 252 prekybos dienos; vidurkio-dispersijos naudingumo rizikos vengimas = {gynybinis_rizikos_vengimas}")
print(f"Statinės vidurkio-dispersijos naudingumo strategijos rizikos vengimas: {bazinis_rizikos_vengimas}")
print(f"Perbalansavimo dažnis: {bazinis_perbalansavimo_daznis}")

# normalizuotų aktyvų kainų grafikas
normalizuotos_kainos = kainos[visata] / kainos[visata].iloc[0] * 100
plt.figure(figsize=(14, 8))
for aktyvas in visata:
    plt.plot(normalizuotos_kainos.index, normalizuotos_kainos[aktyvas], label=aktyvas)

plt.title("Normalizuotos pasirinktų aktyvų kainos")
plt.xlabel("Data")
plt.ylabel("Indekso reikšmė (pradžia = 100)")
plt.legend(ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

# koreliacijos matricos lentelė ir grafikas
koreliaciju_matrica = grazos[visata].corr()
plt.figure(figsize=(11, 9))
plt.imshow(koreliaciju_matrica, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Koreliacija")
plt.xticks(range(len(visata)), visata, rotation=45, ha="right")
plt.yticks(range(len(visata)), visata)

for i in range(len(visata)):
    for j in range(len(visata)):
        reiksme = koreliaciju_matrica.iloc[i, j]
        spalva = "white" if abs(reiksme) > 0.65 else "black"
        plt.text(j, i, f"{reiksme:.2f}", ha="center", va="center", fontsize=8, color=spalva)

plt.title("Aktyvų dieninių grąžų koreliacijos matrica")
plt.tight_layout()
plt.show()
print("\nAKTYVŲ GRĄŽŲ KORELIACIJOS LENTELĖ")
rodyti(koreliaciju_matrica.round(3))

# Rinkos signalų ir režimų svorių grafikas
rinkos_indeksas = spx_sukaupta_graza.copy()
rinkos_grazos = spx_graza.copy()
tendencijos_zaliava = rinkos_indeksas.rolling(50).mean() / rinkos_indeksas.rolling(200).mean() - 1
momento_zaliava = rinkos_indeksas / rinkos_indeksas.shift(252) - 1
kintamumas_20 = rinkos_grazos.rolling(20).std() * np.sqrt(prekybos_dienos)
kintamumas_252 = rinkos_grazos.rolling(252).std() * np.sqrt(prekybos_dienos)
kintamumo_zaliava = kintamumas_20 / kintamumas_252 - 1
tendencijos_z = standartizuota_reiksme(tendencijos_zaliava, z_langas, z_minimalus_stebejimu_kiekis)
momento_z = standartizuota_reiksme(momento_zaliava, z_langas, z_minimalus_stebejimu_kiekis)
kintamumo_z = standartizuota_reiksme(kintamumo_zaliava, z_langas, z_minimalus_stebejimu_kiekis)

rizikos_apetitas = (
    bazinis_tendencijos_svoris * tendencijos_z
    + bazinis_momento_svoris * momento_z
    - bazinis_kintamumo_svoris * kintamumo_z
).clip(-2, 2)

rezimai = pd.DataFrame(
    index=rizikos_apetitas.dropna().index,
    columns=["Agresyvus", "Gynybinis", "Neutralus"],
    dtype=float
)

for data, ra in rizikos_apetitas.dropna().items():
    rezimai.loc[data] = minkstasis_maksimumas([ra, -ra, -abs(ra)], bazine_temperatura)

signalu_lentele = pd.DataFrame({
    "Rinkos indeksas": rinkos_indeksas,
    "Trendo z": tendencijos_z,
    "Momento z": momento_z,
    "Kintamumo z": kintamumo_z,
    "Rizikos apetitas": rizikos_apetitas
}).join(rezimai).dropna()

signalu_lentele = signalu_lentele.loc["2016-01-01":"2026-01-01"]

signalu_statistika = signalu_lentele[
    ["Trendo z", "Momento z", "Kintamumo z", "Rizikos apetitas"]
].agg(["mean", "std", "min", "max"]).T

signalu_statistika.columns = [
    "Vidurkis",
    "Standartinis nuokrypis",
    "Minimali reikšmė",
    "Maksimali reikšmė"
]

rezimu_statistika = signalu_lentele[
    ["Agresyvus", "Gynybinis", "Neutralus"]
].agg(["mean", "std", "min", "max"]).T

rezimu_statistika.columns = [
    "Vidurkis",
    "Standartinis nuokrypis",
    "Minimali reikšmė",
    "Maksimali reikšmė"
]

dominuojantis_rezimas = signalu_lentele[
    ["Agresyvus", "Gynybinis", "Neutralus"]
].idxmax(axis=1)

rezimu_dazniai = dominuojantis_rezimas.value_counts(normalize=True).to_frame("Laikotarpio dalis")
rezimu_pokyciai = (dominuojantis_rezimas != dominuojantis_rezimas.shift(1)).sum() - 1

print("\nRINKOS SIGNALŲ STATISTIKA")
rodyti(signalu_statistika.round(4))

print("\nREŽIMŲ SVORIŲ STATISTIKA")
rodyti(rezimu_statistika.round(4))

print("\nDOMINUOJANČIŲ REŽIMŲ DAŽNIAI")
rodyti(rezimu_dazniai.round(4))

print("\nREŽIMŲ PASIKEITIMŲ SKAIČIUS")
print(rezimu_pokyciai)

print("\nRIZIKOS APETITO TEIGIAMŲ / NEIGIAMŲ DIENŲ DALIS")
print(f"Teigiamas rizikos apetitas: {(signalu_lentele['Rizikos apetitas'] > 0).mean():.2%}")
print(f"Neigiamas rizikos apetitas: {(signalu_lentele['Rizikos apetitas'] < 0).mean():.2%}")

plt.figure(figsize=(14, 6))
plt.plot(signalu_lentele.index, signalu_lentele["Trendo z"], label="Trendas", linewidth=1.2)
plt.plot(signalu_lentele.index, signalu_lentele["Momento z"], label="Momentas", linewidth=1.2)
plt.plot(signalu_lentele.index, signalu_lentele["Kintamumo z"], label="Kintamumas", linewidth=1.2)
plt.plot(signalu_lentele.index, signalu_lentele["Rizikos apetitas"], label="Rizikos apetitas", linewidth=2.4, linestyle="--")
plt.axhline(0, linewidth=1, color="black", alpha=0.7)
plt.title("Rinkos signalai ir rizikos apetito rodiklis")
plt.ylabel("Standartizuota reikšmė")
plt.xlabel("Data")
plt.legend(ncol=4)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.stackplot(
    signalu_lentele.index,
    signalu_lentele["Agresyvus"],
    signalu_lentele["Gynybinis"],
    signalu_lentele["Neutralus"],
    labels=["Agresyvus", "Gynybinis", "Neutralus"],
    alpha=0.9
)
plt.title("Agresyvaus, gynybinio ir neutralaus režimų svoriai")
plt.ylabel("Režimo svoris")
plt.xlabel("Data")
plt.ylim(0, 1)
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# signalų ir režimų vidurkiai metams ir mėnesiams
analizuojami_stulpeliai = [
    "Trendo z",
    "Momento z",
    "Kintamumo z",
    "Rizikos apetitas",
    "Agresyvus",
    "Gynybinis",
    "Neutralus"
]

metiniai_vidurkiai = signalu_lentele[analizuojami_stulpeliai].resample("YE").mean()
metiniai_vidurkiai.index = metiniai_vidurkiai.index.year
menesiniai_vidurkiai = signalu_lentele[analizuojami_stulpeliai].resample("ME").mean()
menesiniai_vidurkiai.index = menesiniai_vidurkiai.index.strftime("%Y-%m")
metinis_rezimu_apibendrinimas = metiniai_vidurkiai.copy()
metinis_rezimu_apibendrinimas["Dominuojantis režimas"] = metiniai_vidurkiai[
    ["Agresyvus", "Gynybinis", "Neutralus"]
].idxmax(axis=1)
menesinis_rezimu_apibendrinimas = menesiniai_vidurkiai.copy()
menesinis_rezimu_apibendrinimas["Dominuojantis režimas"] = menesiniai_vidurkiai[
    ["Agresyvus", "Gynybinis", "Neutralus"]
].idxmax(axis=1)

print("\nMETINIAI SIGNALŲ IR REŽIMŲ VIDURKIAI")
rodyti(metiniai_vidurkiai.round(4))
print("\nMĖNESINIAI SIGNALŲ IR REŽIMŲ VIDURKIAI")
rodyti(menesiniai_vidurkiai.round(4))
print("\nMETINIAI VIDURKIAI SU DOMINUOJANČIU REŽIMU")
rodyti(metinis_rezimu_apibendrinimas.round(4))
print("\nMĖNESINIAI VIDURKIAI SU DOMINUOJANČIU REŽIMU")
rodyti(menesinis_rezimu_apibendrinimas.round(4))

# vidutinių svorių lentelė ir grafikas
vidutiniai_svoriai = pd.DataFrame(index=strategiju_pavadinimai, columns=visata, dtype=float)
for strategija in strategiju_pavadinimai:
    vidutiniai_svoriai.loc[strategija] = svoriu_istorija[strategija].dropna(how="all").mean()

vidutiniai_svoriai = vidutiniai_svoriai.fillna(0).clip(lower=0)
print("\nVIDUTINIAI AKTYVŲ SVORIAI PAGAL STRATEGIJAS")
rodyti(vidutiniai_svoriai.round(4))
vidutiniai_svoriai.plot(kind="bar", stacked=True, figsize=(14, 7), width=0.85)
plt.title("Vidutiniai aktyvų svoriai pagal portfelio strategijas", fontsize=14)
plt.xlabel("Strategija")
plt.ylabel("Vidutinis svoris")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Aktyvas", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# metiniai svoriai
metiniai_visu_strategiju_svoriai = {}
for strategija in strategiju_pavadinimai:
    svoriai = svoriu_istorija[strategija].dropna(how="all").copy()
    svoriai.index = pd.to_datetime(svoriai.index)

    metiniai_svoriai = svoriai.groupby(svoriai.index.year).mean().fillna(0).clip(lower=0)
    metiniai_visu_strategiju_svoriai[strategija] = metiniai_svoriai

    print(f"\nMETINIAI VIDUTINIAI SVORIAI: {strategija}")
    rodyti(metiniai_svoriai.round(4))

statistika = pd.DataFrame({
    "Vidutinė dieninė grąža": grazos.mean(),
    "Dieninis kintamumas": grazos.std(),
    "Min. dieninė grąža": grazos.min(),
    "Max. dieninė grąža": grazos.max(),
    "Metinė grąža": (1 + grazos.mean()) ** 252 - 1,
    "Metinis kintamumas": grazos.std() * np.sqrt(252)
})

print("\nPIRMINĖ STATISTINĖ ANALIZĖ")
rodyti(statistika.round(4))

# portfelių kaupiamosios grąžos grafikas
neto_grazos_lentele = pd.DataFrame(dienines_neto_grazos).dropna(how="any")
kaupiamosios_neto_grazos = (1 + neto_grazos_lentele).cumprod()

plt.figure(figsize=(14, 8))

for strategija in strategiju_pavadinimai:
    plt.plot(kaupiamosios_neto_grazos.index, kaupiamosios_neto_grazos[strategija], label=strategija)

plt.title("Portfelių kaupiamoji grąža po sandorių sąnaudų")
plt.xlabel("Data")
plt.ylabel("Kaupiamoji grąža")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()