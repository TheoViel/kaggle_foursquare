import gc
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
import Levenshtein
import difflib
import sys
import cudf
from numba import jit
from unidecode import unidecode
import re

warnings.simplefilter("ignore")
start_time = time.time()
random.seed(13)

DATA_PATH = "../data/"

df = cudf.read_csv(DATA_PATH + "train.csv").set_index("id")
folds = cudf.read_csv(DATA_PATH + "folds_2.csv")[["id", "fold"]]
df = df.merge(folds, how="left", on="id").set_index("id")
df = df[df["fold"] == 0]
df.sort_index(inplace=True)
train = df.reset_index().to_pandas()


clusts = (
    train[["id", "point_of_interest"]]
    .groupby("point_of_interest")
    .agg(list)
    .reset_index()
    .rename(columns={"id": "matches"})
)

gt = train[["id", "point_of_interest"]].merge(clusts, on="point_of_interest")

N_TO_FIND = gt["matches"].apply(len).sum() - len(gt)
from numerize.numerize import numerize


def print_infos(p1, p2=None):
    p1 = p1.copy()
    if "y" not in p1.columns:
        p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(
            np.int8
        )

    if p2 is not None:
        p2 = p2.copy()

        p2["y"] = p1["y"].values

        p1["id_2"] = p2["id"]
        p2["id_2"] = p1["id"]

        ps = pd.concat([p1, p2], axis=0).reset_index(drop=True)
        ps = ps.drop_duplicates(keep="first", subset=["id", "id_2"])

        ps = ps[ps["id"] != ps["id_2"]]

    else:
        ps = p1

    print(f"Number of candidates : {numerize(len(ps))}")
    print(f"Proportion of positive candidates: {np.sum(ps.y) / len(ps) * 100:.2f}%")
    print(f"Proportion of found matches: {np.sum(ps.y) / N_TO_FIND * 100:.2f}%")




# test specific code
path = "../data/ressources/"


@jit
def lcs2_n(stringA, stringB, matrix):
    substringLength = 0
    for aIndex in range(1, 1 + stringA.shape[0]):
        for bIndex in range(1, 1 + stringB.shape[0]):
            if stringA[aIndex - 1] == stringB[bIndex - 1]:
                t = matrix[aIndex - 1, bIndex - 1] + 1
                if t > substringLength:
                    substringLength = t
                matrix[aIndex, bIndex] = t
    return substringLength


def lcs2(stringA, stringB):
    a = np.frombuffer(stringA.encode(), dtype=np.int8)
    b = np.frombuffer(stringB.encode(), dtype=np.int8)
    matrix = np.zeros([1 + a.shape[0], 1 + b.shape[0]], dtype=np.int8)
    return lcs2_n(a, b, matrix)


@jit
def lcs_n(stringA, stringB, matrix):
    for aIndex in range(1, 1 + stringA.shape[0]):
        for bIndex in range(1, 1 + stringB.shape[0]):
            if stringA[aIndex - 1] == stringB[bIndex - 1]:
                matrix[aIndex, bIndex] = matrix[aIndex - 1, bIndex - 1] + 1
            else:
                matrix[aIndex, bIndex] = max(
                    matrix[aIndex - 1, bIndex], matrix[aIndex, bIndex - 1]
                )
    return matrix[stringA.shape[0], stringB.shape[0]]


def lcs(stringA, stringB):
    a = np.frombuffer(stringA.encode(), dtype=np.int8)
    b = np.frombuffer(stringB.encode(), dtype=np.int8)
    matrix = np.zeros([1 + a.shape[0], 1 + b.shape[0]], dtype=np.int8)
    return lcs_n(a, b, matrix)


#################################################################################
## TF-IDF function
#################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sys import getsizeof


def top_n_idx_sparse(matrix, N_row_pick):
    """
    Renvoie les index des n plus grandes valeurs de chaque ligne d'une sparse matrix
    impose_valeur_differente : Impose (si possible) au moins une valeur non-maximale pour éviter d’ignorer un score maximal si trop d’élèments en ont un
    """

    top_n_idx = []
    i = 0
    # matrix.indptr = index du 1er élèment (non nul) de chaque ligne
    for gauche, droite in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(
            N_row_pick[i], droite - gauche
        )  # pour gérer les cas où n est plus grand que le nombre de valeurs non-nulles de la ligne
        index = matrix.indices[
            gauche
            + np.argpartition(matrix.data[gauche:droite], -n_row_pick)[-n_row_pick:]
        ]
        # Ajout des indexs trouvés
        top_n_idx.append(index[::-1])
        i += 1
    return top_n_idx


def vectorisation_similarite(corpus_A, thr=0.3):
    """Renvoie un dataframe avec les paires de libellés les plus similaires au sens TF-IDF et Jaro-Winkler, entre A et B."""

    # =================================
    # ETAPE 0 : Forçage en string pour éviter les erreurs, suppression des doublons et suppression des espaces en préfixe/suffixe
    corpus_A = [str(x).strip().lower() for x in corpus_A]
    corpus_B = corpus_A.copy()

    # =================================
    # ÉTAPE 1 : Vectorisation du corpus
    vect = TfidfVectorizer()  # min_df=1, stop_words="english"
    tfidf_A = vect.fit_transform(
        corpus_A
    )  # Pas besoin de normaliser par la suite : le Vectorizer renvoie un tf-idf normalisé
    tfidf_B = vect.transform(corpus_B)  # Utilisation de la normalisation issue de A
    pairwise_similarity = (
        tfidf_A * tfidf_B.T
    )  # Sparse matrice (les élèments nuls ne sont pas notés) de dimension égale aux nombres de lignes dans les documents
    N, M = pairwise_similarity.shape  # taille de la matrice

    # =======================================================
    # ÉTAPE 2 : Calcul des indices des n plus grandes valeurs

    # Calcul des élèments non-nuls de pairwise_similarity
    Elt_non_nuls = np.split(
        pairwise_similarity.data[
            pairwise_similarity.indptr[0] : pairwise_similarity.indptr[-1]
        ],
        pairwise_similarity.indptr[1:-1],
    )

    # Calcul du nb d'élèments à checker : tous les bons scores OU les meilleurs scores AVEC au moins nb_best_score
    Nb_elt_max_par_ligne = [
        len(np.argwhere((liste >= thr) | (liste == np.amax(liste))).flatten().tolist())
        if liste.size > 0
        else 0
        for liste in Elt_non_nuls
    ]

    # Taille de la matrice (dense) créée
    taille_MB = round(
        getsizeof(Elt_non_nuls) / (1024 * 1024)
    )  # Taille en MB de la matrixe todense()
    if taille_MB > 10:
        print("  /!\ La taille de la matrice est de {} MB.".format(taille_MB))

    # Calcul des indices argmax dans la csr_matrix pairwise_similarity
    Ind_n_max = top_n_idx_sparse(
        pairwise_similarity, Nb_elt_max_par_ligne
    )  # Calcul des indices des grandes valeurs

    # Récuparéation des valeurs TF-IDF
    Valeurs = []
    for i, Liste_index in enumerate(Ind_n_max):
        values = []
        for idx in Liste_index:
            v = round(pairwise_similarity[(i, idx)], 5)
            values.append(v)

        Valeurs.append(values)

    return Ind_n_max, Valeurs


#################################################################################
## Haversine function
#################################################################################

from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


#################################################################################
#################################################################################


# detect language based on character set
def isEnglish(s):
    ss = "ª°⭐•®’—–™&\xa0\xad\xe2\xf0"  # special characters
    s = str(s).lower()
    for k in range(len(ss)):
        s = s.replace(ss[k], "")
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        # not english; check it still not english if western european characters are removed
        ss = "éáñóüäýöçãõúíàêôūâşè"
        for k in range(len(ss)):
            s = s.replace(ss[k], "")
        try:
            s.encode(encoding="utf-8").decode("ascii")
        except UnicodeDecodeError:
            return 3  # really not english
        else:
            return 2  # spanish/french?
    else:
        return 1  # english


train["lang"] = train["name"].apply(isEnglish).astype("int8")


# fill-in missing categories, based on words in name
Key_words_for_cat = pd.read_pickle(path + "dict_for_missing_cat.pkl")


def process(cat, split=" "):
    cat = [x for x in str(cat).split(split) if cat != "" and len(x) >= 2]
    # Keep only letters
    cat = [re.sub(r"[^a-zA-Z]", " ", x) for x in cat]
    # Delete multi space
    cat = [re.sub("\\s+", " ", x).strip() for x in cat]
    return cat


# Function to fill missing categories
def find_cat(name):
    global Key_words_for_cat
    name_list = process(unidecode(str(name).lower()))
    for cat, wordlist in Key_words_for_cat.items():
        if any(name_word in name_list for name_word in wordlist):
            return cat
    return ""


train["categories"] = train["categories"].fillna("")
idx_missing_cat = train[train["categories"] == ""].index
train.loc[idx_missing_cat, "categories"] = (
    train.loc[idx_missing_cat, "name"].fillna("").apply(find_cat)
)
print(
    "finished filling-in missing categories", int(time.time() - start_time), "sec"
)  # NA cats drop by around 25%
del Key_words_for_cat, idx_missing_cat
gc.collect()


# pre-format data
train["point_of_interest"] = (
    train["point_of_interest"].astype("category").cat.codes
)  # turn POI into ints to save spacetime
train["latitude"] = train["latitude"].astype("float32")
train["longitude"] = train["longitude"].astype("float32")
# sorted by count in candidate training data
countries = [
    "US",
    "ID",
    "TR",
    "JP",
    "TH",
    "RU",
    "MY",
    "BR",
    "SG",
    "PH",
    "BE",
    "KR",
    "GB",
    "MX",
    "DE",
    "FR",
    "ES",
    "CL",
    "UA",
    "IT",
    "CA",
    "AU",
    "SA",
    "CN",
    "HK",
    "FI",
    "NL",
    "TW",
    "AR",
    "GR",
    "CZ",
    "KW",
    "AE",
    "IN",
    "CO",
    "RO",
    "VN",
    "IR",
    "HU",
    "SE",
    "PE",
    "PL",
    "LV",
    "PT",
    "EG",
    "AT",
    "ZA",
    "CH",
    "BY",
    "PY",
    "RS",
    "CR",
    "DK",
    "BG",
    "IE",
    "VE",
    "DO",
    "MV",
    "CY",
    "MK",
    "EE",
    "NZ",
    "PR",
    "BN",
    "HR",
    "NO",
    "SK",
    "IL",
    "EC",
    "MD",
    "PA",
    "LT",
    "GT",
    "KH",
    "QA",
    "BH",
    "AZ",
    "GE",
    "SV",
    "TN",
    "LK",
    "JO",
    "UY",
    "KE",
    "KZ",
    "MQ",
    "LB",
    "MA",
    "IS",
    "HN",
    "SI",
    "MT",
    "GU",
    "ME",
    "OM",
    "BO",
    "TT",
    "LU",
    "JM",
    "PK",
    "BD",
    "XX",
    "MN",
    "AM",
    "TM",
    "MO",
    "LA",
    "NI",
    "BA",
    "KG",
    "NG",
    "BB",
    "UZ",
    "NP",
    "BS",
    "GH",
    "AW",
    "AL",
    "TZ",
    "IQ",
    "IM",
    "UG",
    "MU",
    "VI",
    "MC",
    "GP",
    "SD",
    "XK",
    "KY",
    "ZM",
    "MP",
    "MZ",
    "AX",
    "CU",
    "BM",
    "SY",
    "ET",
    "JE",
    "BZ",
    "SM",
    "LC",
    "GG",
    "VA",
    "BL",
    "TC",
    "CW",
    "AD",
    "RE",
    "PF",
    "SR",
    "GI",
    "BW",
    "AO",
    "HT",
    "FJ",
    "WS",
    "GD",
    "GF",
    "KP",
    "VC",
    "RW",
    "SC",
    "MG",
    "DZ",
    "VG",
    "SX",
    "PS",
    "AF",
    "MF",
    "AG",
    "CM",
    "CI",
    "DM",
    "CD",
    "SN",
    "LI",
    "AQ",
    "MW",
    "TL",
    "BT",
    "CV",
    "KN",
    "BF",
    "AI",
    "SZ",
    "ZW",
    "AN",
    "LY",
    "NC",
    "YE",
    "SO",
    "GA",
    "EU",
    "PM",
    "BI",
    "GL",
    "GM",
    "BV",
    "NE",
    "GW",
    "TJ",
    "BQ",
    "GQ",
    "BJ",
    "TG",
    "ST",
    "VU",
    "PG",
    "PW",
    "TO",
    "SH",
    "GY",
    "SL",
    "YT",
    "FO",
    "DJ",
    "EH",
    "SJ",
    "LR",
    "SS",
    "ZZ",
]
c_di = {}
for i, c in enumerate(countries):  # map train/test countries the same way
    c_di[c] = min(
        50, i + 1
    )  # cap country at 50 - after that there are too few cases per country to split them
train["country"] = (
    train["country"].fillna("ZZ").map(c_di).fillna(50).astype("int16")
)  # new country maps to missing (ZZ)


# true groups - id's of the same POI; this is the answer we are trying to predict; use it for CV
train = train.reset_index()
train = train.sort_values(by=["point_of_interest", "id"]).reset_index(drop=True)
id_all = np.array(train["id"])
poi_all = np.array(train["point_of_interest"])
poi0 = poi_all[0]
id0 = id_all[0]
di_poi = {}
for i in range(1, train.shape[0]):
    if poi_all[i] == poi0:
        id0 = str(id0) + " " + str(id_all[i])
    else:
        di_poi[poi0] = str(id0) + " "  # need to have trailing space in m_true
        poi0 = poi_all[i]
        id0 = id_all[i]
di_poi[poi0] = str(id0) + " "  # need to have trailing space in m_true
train["m_true"] = train["point_of_interest"].map(di_poi)
train = train.sort_values(by="index").reset_index(
    drop=True
)  # sort back to original order
train.drop("index", axis=1, inplace=True)
print("finished true groups", int(time.time() - start_time), "sec")
a = train.groupby("point_of_interest").size().reset_index()
print(
    "count of all matching pairs is:", (a[0] - 1).sum()
)  # match:398,786 - minimum # of pairs to get correct result
print(
    "count2 of all matching pairs is:",
    (a[0] * (a[0] - 1) // 2).sum(),
    int(time.time() - start_time),
    "sec",
)  # max # of pairs, including overlaps
del a, id_all, poi_all, di_poi
gc.collect()


# create grouped category - column 'category_simpl' ****************************************

# Save copy
train["name_svg"] = train["name"].copy()
train["categories_svg"] = train["categories"].copy()

# Clean name
train["name"] = train["name"].apply(lambda x: unidecode(str(x).lower()))


def replace_seven_eleven(text):
    new = "seven eleven"
    for sub in ["7/11", "7-11", "7-eleven"]:
        text = text.replace(sub + "#", new + " ")
        text = text.replace(sub + " ", new + " ")
        text = text.replace(sub, new)
    return text


train["name"] = train["name"].apply(lambda text: replace_seven_eleven(text))


def replace_seaworld(text):
    new = "seaworld"
    for sub in ["sea world"]:
        text = text.replace(sub, new)
    return text


train["name"] = train["name"].apply(lambda text: replace_seaworld(text))


def replace_mcdonald(text):
    new = "mac donald"
    for sub in [
        "mc donald",
        "mcd ",
        "macd ",
        "mcd",
        "mcdonald",
        "macdonald",
        "mc donalds",
        "mac donalds",
    ]:
        text = text.replace(sub, new)
    return text


train["name"] = train["name"].apply(lambda text: replace_mcdonald(text))

# Grouped categories
Cat_regroup = [
    [
        "airport terminals",
        "airports",
        "airport services",
        "airport lounges",
        "airport food courts",
        "airport ticket counter",
        "airport trams",
        "airfields",
    ],
    ["bus stations", "bus stops"],
    ["opera houses", "concert halls"],
    ["metro stations", "tram stations", "light rail stations", "train stations"],
    [
        "auto garages",
        "auto workshops",
        "automotive shops",
        "auto dealerships",
        "motorcycle shops",
        "new auto dealerships",
    ],
    [
        "hotels",
        "casinos",
        "hotel bars",
        "motels",
        "resorts",
        "residences",
        "inns",
        "hostels",
        "bed breakfasts",
    ],
    [
        "bakeries",
        "borek places",
        "cupcake shops",
        "bagel shops",
        "breakfast spots",
        "gozleme places",
    ],
    [
        "college classrooms",
        "college labs",
        "college science buildings",
        "college arts buildings",
        "college history buildings",
        "college cricket pitches",
        "college communications buildings",
        "college academic buildings",
        "college quads",
        "college auditoriums",
        "college engineering buildings",
        "college math buildings",
        "college bookstores",
        "college technology buildings",
        "college libraries",
        "libraries",
        "college football fields",
        "college administrative buildings",
        "general colleges universities",
        "universities",
        "community colleges",
        "high schools",
        "student centers",
        "college residence halls",
        "schools",
        "private schools",
    ],
    ["movie theaters", "film studios", "indie movie theaters", "multiplexes"],
    [
        "emergency rooms",
        "hospitals",
        "medical centers",
        "hospital wards",
        "medical supply stores",
        "physical therapists",
        "maternity clinics",
        "medical labs",
        "doctor s offices",
    ],
    ["baggage claims", "general travel", "toll plazas"],
    [
        "cafes",
        "dessert shops",
        "donut shops",
        "coffee shops",
        "ice cream shops",
        "corporate coffee shops",
        "coffee roasters",
    ],
    ["hockey arenas", "basketball stadiums", "hockey fields", "hockey rinks"],
    [
        "buildings",
        "offices",
        "coworking spaces",
        "insurance offices",
        "banks",
        "campaign offices",
        "trailer parks",
        "atms",
    ],
    ["capitol buildings", "government buildings", "police stations"],
    ["beach bars", "beaches", "surf spots", "nudist beaches"],
    [
        "asian restaurants",
        "shabu shabu restaurants",
        "noodle houses",
        "chinese restaurants",
        "malay restaurants",
        "sundanese restaurants",
        "cantonese restaurants",
        "chinese breakfast places",
        "ramen restaurants",
        "indonesian restaurants",
        "satay restaurants",
        "javanese restaurants",
        "padangnese restaurants",
        "indonesian meatball places",
    ],
    [
        "historic sites",
        "temples",
        "mosques",
        "spiritual centers",
        "monasteries",
        "churches",
        "history museums",
        "buddhist temples",
        "mountains",
    ],
    ["bars", "pubs"],
    [
        "gyms",
        "gyms or fitness centers",
        "gymnastics gyms",
        "gym pools",
        "yoga studios",
        "badminton courts",
        "courthouses",
    ],
    ["fast food restaurants", "burger joints", "fried chicken joints"],
    [
        "nail salons",
        "salons barbershops",
        "perfume shops",
        "department stores",
        "cosmetics shops",
    ],
    [
        "alternative healers",
        "health beauty services",
        "chiropractors",
        "acupuncturists",
    ],
    ["grocery stores", "health food stores", "supermarkets"],
    ["boutiques", "clothing stores"],
    ["elementary schools", "middle schools"],
    ["electronics stores", "mobile phone shops"],
    ["convenience stores", "truck stops", "gas stations"],
    ["theme park rides attractions", "theme parks"],
    ["outlet malls", "shopping malls", "adult boutiques", "shopping plazas"],
    ["farmers markets", "markets"],
    ["general entertainment", "paintball fields"],
    ["som tum restaurants", "thai restaurants"],
    ["piers", "ports"],
    ["rugby stadiums", "soccer stadiums", "stadiums", "soccer fields"],
    ["lounges", "vape stores"],
    ["massage studios", "spas"],
    ["racecourses", "racetracks"],
    ["men s stores", "women s stores"],
    ["american restaurants", "tex mex restaurants"],
    ["japanese restaurants", "sushi restaurants"],
    ["indian restaurants", "mamak restaurants"],
    ["baseball fields", "baseball stadiums"],
    ["tennis courts", "tennis stadiums"],
    ["drugstores", "pharmacies"],
    ["city halls", "town halls"],
    ["ski areas", "ski chalets", "ski lodges"],
    ["lakes", "reservoirs"],
    ["greek restaurants", "tavernas"],
    ["hills", "scenic lookouts"],
    [
        "college soccer fields",
        "college stadiums",
        "college hockey rinks",
        "college tracks",
        "college basketball courts",
    ],
    ["furniture home stores", "mattress stores", "lighting stores"],
    ["recruiting agencies", "rehab centers"],
    ["art museums", "art studios", "art galleries", "museums", "history museums"],
    ["outdoor supply stores", "sporting goods shops"],
    ["czech restaurants", "restaurants"],
    ["street fairs", "street food gatherings"],
    ["canal locks", "canals"],
    ["sake bars", "soba restaurants"],
    ["bookstores", "newsagents", "newsstands", "stationery stores"],
    ["other great outdoors", "rafting spots"],
    ["manti places", "turkish restaurants"],
    ["shoe repair shops", "shoe stores"],
    ["photography labs", "photography studios"],
    ["bowling alleys", "bowling greens"],
    ["dry cleaners", "laundry services"],
    ["cigkofte places", "kofte places"],
    ["strip clubs", "other nightlife", "gay bars", "nightclubs", "rock clubs"],
    ["dog runs", "parks", "forests", "rv parks", "playgrounds"],
    ["convention centers", "event spaces", "conventions"],
    ["cruise ships", "harbors marinas", "piers", "boats or ferries"],
    ["italian restaurants", "pizza places"],
    ["law schools", "lawyers"],
    ["bubble tea shops", "tea rooms"],
    ["monuments landmarks", "outdoor sculptures"],
    ["beer bars", "beer stores", "beer gardens", "breweries", "brasseries"],
    ["kebab restaurants", "steakhouses"],
    [
        "real estate offices",
        "rental services",
        "rental car locations",
        "residential buildings apartments condos",
    ],
    ["golf courses", "mini golf courses"],
    ["food drink shops", "food services", "food stands", "food trucks"],
    ["salad places", "sandwich places", "shawarma places"],
    ["ski chairlifts", "ski trails", "apres ski bars", "skate parks"],
    ["wine shops", "wineries"],
    ["flea markets", "floating markets"],
    ["burrito places", "taco places"],
    ["pet services", "pet stores", "veterinarians"],
    ["music festivals", "music venues", "music stores", "music schools"],
    ["irish pubs", "pie shops"],
    ["zoo exhibits", "exhibits", "zoos"],
    ["general travel", "bridges"],
    ["sporting goods shops", "athletics & sports", "hunting supplies"],
    ["optical shops", "eye doctors"],
    ["home services & repairs", "other repair shops"],
]

import re


def process_text(text):
    text = unidecode(text.lower())
    res = " ".join([re.sub(r"[^a-zA-Z]", " ", x).strip() for x in text.split()])
    return re.sub("\\s+", " ", res).strip()


def simplify_cat(categories):
    global Cat_regroup
    categories = str(categories).lower()
    if categories in ("", "nan"):
        return -1
    for cat in categories.split(","):
        cat = process_text(cat)
        for i, Liste in enumerate(Cat_regroup):
            if any(cat == x for x in Liste):
                return i + 1
    else:
        return 0


train["category_simpl"] = (
    train["categories"]
    .astype(str)
    .apply(lambda text: simplify_cat(text))
    .astype("int16")
)

print(
    "Simpl categories found :", len(train[train["category_simpl"] > 0]), "/", len(train)
)

# Go back to initial columns
train["name"] = train["name_svg"].copy()
train["categories"] = train["categories_svg"].copy()
train.drop(["name_svg", "categories_svg"], axis=1, inplace=True)


# remove all spaces, symbols, lower case
def st(x, remove_space=False):
    # turn to latin alphabet
    x = unidecode(str(x))
    # lower case
    x = x.lower()
    # remove symbols
    x = x.replace('"', "")
    ss = ",:;'/-+&()!#$%*.|\@`~^<>?[]{}_=\n"
    if remove_space:
        ss = " " + ss
    for i in range(len(ss)):
        x = x.replace(ss[i], "")
    return x


def st2(x):  # remove numbers - applies to cities only
    ss = " 0123456789"
    for i in range(len(ss)):
        x = x.replace(ss[i], "")
    return x


# Save names separated by spaces for tf-idf
train["categories_split"] = (
    train["categories"]
    .astype(str)
    .apply(lambda x: [st(cat, remove_space=True) for cat in x.split(",")])
    .copy()
)  # Create a new columns to split the categories
train["name_initial"] = train["name"].astype(str).apply(lambda x: x.lower()).copy()
train["name_initial_decode"] = (
    train["name"].astype(str).apply(lambda x: st(x, remove_space=False)).copy()
)

solo_cat_scores = pd.read_pickle(
    path + "howmanytimes_groupedcat_are_paired_with_other_groupedcat.pkl"
)  # link-between-grouped-cats

# Find the score of the categories
train["freq_pairing_with_other_groupedcat"] = (
    train["category_simpl"].apply(lambda cat: solo_cat_scores[cat]).fillna(0)
)

solo_cat_scores = pd.read_pickle(
    path + "solo_cat_score.pkl"
)  # link-between-categories - 1858 values


def apply_solo_cat_score(List_cat):
    # global solo_cat_scores
    return max([solo_cat_scores[cat] for cat in List_cat])


# Find the score of the categories
train["cat_solo_score"] = (
    train["categories_split"]
    .apply(lambda List_cat: apply_solo_cat_score(List_cat))
    .fillna(0)
)

Dist_quantiles = pd.read_pickle(
    path + "Dist_quantiles_per_cat.pkl"
)  # dist-quantiles-per-cat - 869 values


def apply_cat_distscore(List_cat):
    # global Dist_quantiles
    q = np.array([Dist_quantiles[cat] for cat in List_cat if cat in Dist_quantiles])
    if len(q) == 0:
        return Dist_quantiles[""]
    return np.max(q, axis=0)


# Find the scores
col_cat_distscores = ["Nb_multiPoi", "mean", "q25", "q50", "q75", "q90", "q99"]
train.loc[:, col_cat_distscores] = (
    train["categories_split"].apply(apply_cat_distscore).to_list()
)  # 'Nb_multiPoi', 'mean', 'q25', 'q50', 'q75', 'q90','q99'
for col in [
    "cat_solo_score",
    "freq_pairing_with_other_groupedcat",
    "Nb_multiPoi",
    "mean",
    "q25",
    "q50",
    "q75",
    "q90",
    "q99",
]:
    train[col] = train[col].astype("float32")


# remove some expressions from name*********************************************
def rem_expr(x):
    x = str(x)
    x = x.replace("™", "")  # tm
    x = x.replace("®", "")  # r
    x = x.replace("ⓘ", "")  # i
    x = x.replace("©", "")  # c
    return x


train["name"] = train["name"].apply(rem_expr)

# drop abbreviations all caps in brakets for long enough names*******************
def rem_abr(x):
    x = str(x)
    if "(" in x and ")" in x:  # there are brakets
        i = x.find("(")
        j = x.find(")")
        if (
            j > i + 1 and j - i < 10 and len(x) - (j - i) > 9
        ):  # remainder is long enough
            s = x[i + 1 : j]
            # clean it
            ss = " ,:;'/-+&()!#$%*.|`~^<>?[]{}_=\n"
            for k in range(len(ss)):
                s = s.replace(ss[k], "")
            if s == s.upper():  # all caps (and/or numbers)
                x = x[:i] + x[j + 1 :]
    return x


train["name"] = train["name"].apply(rem_abr)


def clean_nums(x):  # remove st/nd/th number extensions
    words = [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "9th",
        "0th",
        "1th",
        "2th",
        "3th",
        "4 th",
        "5 th",
        "6 th",
        "7 th",
        "8 th",
        "9 th",
        "0 th",
        "1 th",
        "2 th",
        "3 th",
        "1 st",
        "2 nd",
        "3 nd",
    ]
    for word in words:
        x = x.replace(word, word[0])
    return x


def rem_words(x):  # remove common words without much meaning
    words = [
        "the",
        "de",
        "of",
        "da",
        "la",
        "a",
        "an",
        "and",
        "at",
        "b",
        "el",
        "las",
        "los",
        "no",
        "di",
        "by",
        "le",
        "del",
        "in",
        "co",
        "inc",
        "llc",
        "llp",
        "ltd",
        "on",
        "der",
        " das",
        "die",
    ]
    for word in words:
        x = x.replace(" " + word + " ", " ")  # middle
        if x[: len(word) + 1] == word + " ":  # start
            x = x[len(word) + 1 :]
        if x[-len(word) - 1 :] == " " + word:  # end
            x = x[: -len(word) - 1]
    return x


# select capitals only, or first letter of each word (which could have been capital)
def get_caps_leading(name):
    name = unidecode(name)
    if name[:3].lower() == "the":  # drop leading 'the' - do not include it in nameC
        name = name[3:]
    name = rem_words(
        name
    )  # remove common words without much meaning; assume they are always lowercase
    name = clean_nums(name)  # remove st/nd/th number extensions
    name = [x for x in str(name).split(" ") if name != "" and len(x) >= 2]
    # keep only capitals or first letters
    name = [re.findall(r"^[a-z]|[A-Z]", x) for x in name]
    # merge
    name = ["".join(x) for x in name]
    name = "".join(name)
    return name.lower()


train["nameC"] = train["name"].fillna("").apply(get_caps_leading)


def clean_address(x):
    wwords = [
        ["str", "jalan", "jl", "st", "street", "ul", "ulitsa", "rue", "rua", "via"],
        ["rd", "road"],
        ["ave", "av", "avenue", "avenida"],
        ["hwy", "highway"],
        ["fl", "floor", "flr"],
        ["blvd", "boulevard", "blv"],
        ["center", "centre"],
        ["dr", "drive"],
        ["mah", "mahallesi"],
        ["ste", "suite"],
        ["prosp", "prospekt"],
    ]
    for words in wwords:
        for word in words[1:]:
            x = x.replace(" " + word + " ", " " + words[0] + " ")  # middle
            if x[: len(word) + 1] == word + " ":  # start
                x = x.replace(word + " ", words[0] + " ")
            if x[-len(word) - 1 :] == " " + word:  # end
                x = x.replace(" " + word, " " + words[0])
    return x


for col in ["name", "address", "city", "state", "zip", "url", "categories"]:
    train[col] = train[col].astype("str").apply(st)  # keep spaces
    if col in ["name", "address"]:
        train[col] = train[col].apply(rem_words)
        train[col] = train[col].apply(clean_nums)
        if col == "address":
            train["address"] = train["address"].apply(clean_address)
    train[col] = train[col].apply(lambda x: x.replace(" ", ""))  # remove spaces

train["city"] = train["city"].apply(st2)  # remove digits from cities
train["latitude"] = np.round(train["latitude"], 5).astype("float32")
train["longitude"] = np.round(train["longitude"], 5).astype("float32")
# for sorting - rounded coordinates
train["lat2"] = np.round(train["latitude"], 0).astype("float32")
train["lon2"] = np.round(train["longitude"], 0).astype("float32")
# for sorting - short name
train["name2"] = train["name"].str[:7]
print("finished pre-processing", int(time.time() - start_time), "sec")


# support functions**************************************************************************************
# distance, in meters
def distance(lat1, lon1, lat2, lon2):
    lat1 = lat1 * 3.14 / 180.0
    lon1 = lon1 * 3.14 / 180.0
    lat2 = lat2 * 3.14 / 180.0
    lon2 = lon2 * 3.14 / 180.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 6373000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return distance


# partial intersection.
def pi(x, y):  # is starting substring of x in y?
    m = min(len(x), len(y))
    for l in range(m, 0, -1):
        if y[:l] in x or x[:l] in y:
            return l
    return 0


def pi1(x, y):  # pi=partial intersection: check if first N letters are in the other
    if y[:4] in x or x[:4] in y:  # hardcode 4 here - for now
        return 1
    else:
        return 0


def pi2(x, y):  # is ending substring of x in y?
    m = min(len(x), len(y))
    for l in range(m, 0, -1):
        if y[-l:] in x or x[-l:] in y:
            return l
    return 0


# clean name
name_di = {
    "uluslararası": "international",
    "havaalani": "airport",
    "havalimani": "airport",
    "stantsiiametro": "metro",
    "aeropuerto": "airport",
    "seveneleven": "7eleven",
    "kfckfc": "kfc",
    "carefour": "carrefour",
    "makdonalds": "mcdonalds",
    "xingbake": "starbucks",
    "mcdonaldss": "mcdonalds",
    "kentuckyfriedchicken": "kfc",
    "restoran": "restaurant",
    "aiport": "airport",
    "terminal": "airport",
    "starbuckscoffee": "starbucks",
    "7elevenechewniielfewn": "7eleven",
    "adtsecurityservices": "adt",
    "ambarrukmoplaza": "ambarukmoplaza",
    "attauthorizedretailer": "att",
    "bandarjakarta": "bandardjakarta",
    "dairyqueenaedriikhwiin": "dairyqueen",
    "dunkindonut": "dunkin",
    "dunkindonuts": "dunkin",
    "dunkindonutsdnkndwnts": "dunkin",
    "ionmoruguichuanzhuchechang": "ionmoruguichuan",
    "tebingkaraton": "tebingkeraton",
    "tebingkraton": "tebingkeraton",
    "711": "7eleven",
    "albertsonspharmacy": "albertsons",
    "applebeesgrillbar": "applebees",
    "attstore": "att",
    "autozoneautoparts": "autozone",
    "awrestaurant": "aw",
    "chilisgrillbar": "chilis",
    "creditrepairservices": "creditrepair",
    "dominospizza": "dominos",
    "firestonecompleteautocare": "firestone",
    "flyingjtravelcenter": "flyingj",
    "libertytaxservice": "libertytax",
    "mcdonald": "mcdonalds",
    "papajohnspizza": "papajohns",
    "pepboysautopartsservice": "pepboys",
    "piatiorochka": "piaterochka",
    "pilottravelcenters": "pilottravelcenter",
    "sainsburyslocal": "sainsburys",
    "sberbankrossii": "sberbank",
    "shellgasstation": "shell",
    "sprintstore": "sprint",
    "strbks": "starbucks",
    "starbucksreserve": "starbucks",
    "usbankbranch": "usbank",
    "verizonauthorizedretailercellularsales": "verizon",
    "verizonwireless": "verizon",
    "vodafonecepmerkezi": "vodafone",
    "vodafoneshop": "vodafone",
    "walmartneighborhoodmarket": "walmart",
    "walmartsupercenter": "walmart",
    "wellsfargobank": "wellsfargo",
    "zaxbyschickenfingersbuffalowings": "zaxbys",
    "ashleyfurniturehomestore": "ashleyhomestore",
    "ashleyfurniture": "ashleyhomestore",
}
# fix some misspellings
for w in name_di.keys():
    train["name"] = train["name"].apply(lambda x: x.replace(w, name_di[w]))

# new code from V *************************************************************
# Group names
name_groups = pd.read_pickle(path + "name_groups.pkl")
# Translation
trans = {}
for best, group in name_groups.items():
    for n in group:
        trans[n] = best
train["name_grouped"] = train["name"].apply(lambda n: trans[n] if n in trans else n)
print(
    f"Grouped names : {len(train[train['name_grouped'] != train['name']])}/{len(train)}."
)
train["name"] = train["name_grouped"].copy()
train = train.drop(columns=["name_grouped"])
del name_groups, trans
gc.collect()

# cap length at 76
train["name"] = train["name"].str[:76]
# eliminate some common words that do not change meaning
for w in ["center"]:
    train["name"] = train["name"].apply(lambda x: x.replace(w, ""))
train["name"].loc[train["name"] == "nan"] = ""
# walmart
train["name"] = train["name"].apply(lambda x: "walmart" if "walmart" in x else x)
# carrefour
train["name"] = train["name"].apply(lambda x: "carrefour" if "carrefour" in x else x)
# drop leading 'the' from name
idx = train["name"].str[:3] == "the"  # happens 17,712 times = 1.5%
train["name"].loc[idx] = train["name"].loc[idx].str[3:]
print("finished cleaning names", int(time.time() - start_time), "sec")

# clean city
city_di = {
    "alkhubar": "alkhobar",
    "khobar": "alkhobar",
    "muratpasa": "antalya",
    "antwerpen": "antwerp",
    "kuta": "badung",
    "bandungregency": "bandung",
    "bengaluru": "bangalore",
    "bkk": "bangkok",
    "krungethphmhaankhr": "bangkok",
    "pattaya": "banglamung",
    "sathon": "bangrak",
    "silom": "bangrak",
    "beijingshi": "beijing",
    "beograd": "belgrade",
    "ratchathewi": "bangkok",
    "brussels": "brussel",
    "bruxelles": "brussel",
    "bucuresti": "bucharest",
    "capitalfederal": "buenosaires",
    "busangwangyeogsi": "busan",
    "cagayandeorocity": "cagayandeoro",
    "cebucity": "cebu",
    "mueangchiangmai": "chiangmai",
    "qianxieshi": "chiba",
    "qiandaitianqu": "chiyoda",
    "zhongyangqu": "chuo",
    "sumedang": "cikeruh",
    "mexico": "ciudaddemexico",
    "mexicocity": "ciudaddemexico",
    "mexicodf": "ciudaddemexico",
    "koln": "cologne",
    "kobenhavn": "copenhagen",
    "osaka": "dabanshibeiqu",
    "jakarta": "dkijakarta",
    "dnipropetrovsk": "dnepropetrovsk",
    "frankfurtammain": "frankfurt",
    "fukuoka": "fugangshi",
    "minato": "gangqu",
    "moscow": "gorodmoskva",
    "moskva": "gorodmoskva",
    "sanktpeterburg": "gorodsanktpeterburg",
    "spb": "gorodsanktpeterburg",
    "hoankiem": "hanoi",
    "yokohama": "hengbangshi",
    "hochiminhcity": "hochiminh",
    "shouye_": "home_",
    "krungethph": "huaikhwang",
    "konak": "izmir",
    "kocaeli": "izmit",
    "jakartacapitalregion": "dkijakarta",
    "southjakarta": "jakartaselatan",
    "shanghai": "jingan",
    "shanghaishi": "jingan",
    "kyoto": "jingdushi",
    "melikgazi": "kayseri",
    "kharkov": "kharkiv",
    "kiyiv": "kiev",
    "kyiv": "kiev",
    "paradise": "lasvegas",
    "lisbon": "lisboa",
    "makaticity": "makati",
    "mandaluyongcity": "mandaluyong",
    "milano": "milan",
    "mingguwushizhongqu": "mingguwushi",
    "nagoyashi": "mingguwushi",
    "munich": "munchen",
    "muntinlupacity": "muntinlupa",
    "pasaycity": "pasay",
    "pasigcity": "pasig",
    "samsennai": "phayathai",
    "praha": "prague",
    "santiagodechile": "santiago",
    "zhahuangshi": "sapporo",
    "seoulteugbyeolsi": "seoul",
    "shenhushizhongyangqu": "shenhushi",
    "shibuya": "shibuiguqu",
    "xinsuqu": "shinjuku",
    "sofiia": "sofia",
    "surakarta": "solo",
    "suwonsi": "suweonsi",
    "taguigcity": "taguig",
    "taipei": "taibeishi",
    "watthana": "vadhana",
    "wien": "vienna",
    "warszawa": "warsaw",
    "washingtondc": "washington",
    "surgutkhantymansiiskiiavtonomnyiokrugiugraaorossiiskaiafederatsiia": "surgut",
    "newyorkcity": "newyork",
    "newyorknyus": "newyork",
    "ny": "newyork",
    "nyc": "newyork",
    "londongreaterlondon": "london",
    "greaterlondon": "london",
    "losangelescaus": "losangeles",
    "dabanshibeiqu": "dabanshi",
    "seoulsi": "seoul",
    "kuwaitcity": "kuwait",
    "bangkoknoi": "bangkok",
}
for key in city_di.keys():
    train["city"].loc[train["city"] == key] = city_di[key]
# second pass
city_di2 = {
    "jakartaselatan": "dkijakarta",
    "jakartapusat": "dkijakarta",
    "jakartabarat": "dkijakarta",
    "jakartautara": "dkijakarta",
    "jakartatimur": "dkijakarta",
    "saintpetersburg": "sanktpeterburg",
}
for key in city_di2.keys():
    train["city"].loc[train["city"] == key] = city_di2[key]

# cap length at 38
train["city"] = train["city"].str[:38]
# eliminate some common words that do not change meaning
for w in ["gorod"]:
    train["city"] = train["city"].apply(lambda x: x.replace(w, ""))
train["city"].loc[train["city"] == "nan"] = ""
print("finished cleaning cities", int(time.time() - start_time), "sec")

# clean address
train["address"].loc[train["address"] == "nan"] = ""
# cap length at 99
train["address"] = train["address"].str[:99]
train["address"] = train["address"].apply(lambda x: x.replace("street", "str"))

# clean state
# cap length at 33
train["state"] = train["state"].str[:33]
state_di = {
    "calif": "ca",
    "jakartacapitalregion": "jakarta",
    "moscow": "moskva",
    "seoulteugbyeolsi": "seoul",
}
for key in state_di.keys():
    train["state"].loc[train["state"] == key] = state_di[key]
train["state"].loc[train["state"] == "nan"] = ""

# clean url
# cap length at 129
train["url"] = train["url"].str[:129]
train["url"].loc[train["url"] == "nan"] = ""
idx = train["url"].str[:8] == "httpswww"
train["url"].loc[idx] = train["url"].str[8:].loc[idx]
idx = train["url"].str[:7] == "httpwww"
train["url"].loc[idx] = train["url"].str[7:].loc[idx]
idx = train["url"].str[:5] == "https"
train["url"].loc[idx] = train["url"].str[5:].loc[idx]
idx = train["url"].str[:4] == "http"
train["url"].loc[idx] = train["url"].str[4:].loc[idx]
train["url"].loc[train["url"] == "nan"] = ""

# clean phone
def process_phone(text):
    text = str(text)
    if text == "nan":
        return ""
    L = []
    for char in text:
        if char.isdigit():
            L.append(char)
    res = "".join(L)[-10:].zfill(10)
    if len(res) > 0:
        return res
    else:
        return text


train["phone"] = train["phone"].apply(lambda text: process_phone(text))
# all matches of last 9 digits look legit - drop leading digit
train["phone"] = train["phone"].str[1:]
# set invalid numbers to empty
idx = (train["phone"] == "000000000") | (train["phone"] == "999999999")
train["phone"].loc[idx] = ""

# clean categories
# cap length at 68
train["categories"] = train["categories"].str[:68]
train["categories"].loc[train["categories"] == "nan"] = ""
cat_di = {"aiport": "airport", "terminal": "airport"}
for key in cat_di.keys():
    train["categories"] = train["categories"].apply(
        lambda x: x.replace(key, cat_di[key])
    )
print("finished cleaning categories", int(time.time() - start_time), "sec")


# translate some common words*******************************************************************************************************
# indonesia
idx = train["country"] == 2  # ID
id_di = {
    "bandarudara": "airport",
    "bandara": "airport",
    "ruang": "room",
    "smanegeri": "sma",
    "sman": "sma",
    "gedung": "building",
    "danau": "lake",
    "sumatera": "sumatra",
    "utara": "north",
    "barat": "west",
    "jawa": "java",
    "timur": "east",
    "tengah": "central",
    "selatan": "south",
    "kepulauan": "island",
}


def id_translate(x):  # translate, and move some new words to the beginning
    global id_di
    for k in id_di.keys():
        if k in x:
            if id_di[k] in [
                "north",
                "west",
                "east",
                "central",
                "south",
            ]:  # these go in the front
                x = id_di[k] + x.replace(k, "")
            elif id_di[k] in ["building", "lake", "island"]:  # these go in the back
                x = x.replace(k, "") + id_di[k]
            else:
                x = x.replace(k, id_di[k])
    return x


for col in ["name", "address", "city", "state"]:
    train[col].loc[idx] = train[col].loc[idx].apply(id_translate)
print("finished translating ID", int(time.time() - start_time), "sec")


# translate russian words
dict_ru_en = pd.read_pickle(path + "dict_translate_russian.pkl")


def process_text(x):
    return re.findall("\w+", x.lower().strip())


def translate_russian_word_by_word(text):
    global dict_ru_en
    text = process_text(text)
    text = [dict_ru_en[word] if word in dict_ru_en else word for word in text]
    return " ".join(text)


idx = train["country"] == 6  # RU
for k in ["city", "state", "address", "name"]:
    train.loc[idx, k] = (
        train.loc[idx, k].astype(str).apply(translate_russian_word_by_word)
    )
    train.loc[idx, k] = train.loc[idx, k].apply(lambda x: "" if x == "nan" else x)
del dict_ru_en


# match some identical names - based on analysis of mismatched names for true pairs
# soekarno-hatta international airport - Jakarta, ID
l1 = [
    "soekarnohattainternationalairport",
    "soekarnohatta",
    "soekarnohattaairport",
    "airportsoekarnohatta",
    "airportinternasionalsoekarnohatta",
    "soekarnohattainternationalairportjakarta",
    "airportsoekarnohattajakarta",
    "airportinternationalsoekarnohatta",
    "internationalairportsoekarnohatta",
    "soekarnohattaintlairport",
    "airportsoetta",
    "soettainternationalairport",
    "soekarnohattainternasionalairport",
    "soekarnohattaairport",
    "soekarnohattaairport",
    "soetta",
    "airportsukarnohatta",
    "soekarnohattaintairport",
    "soekarnohattaairportinternational",
    "soekarnohattaairportjakarta",
    "airportsoekarno",
    "jakartainternationalairportsoekarnohatta",
    "airportsoekarnohattaindonesia",
    "airportsukarnohattainternational",
    "soekarnohattainternationalairportckg",
]
idx = train["country"] == 2  # ID - this is where this location is
train["name"].loc[idx] = (
    train["name"].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
)
l1 = [
    "kualanamuinternationalairport",
    "kualanamuairport",
    "airportkualanamu",
    "kualanamuinternationalairportmedan",
    "airportinternasionalkualanamu",
    "kualanamointernationalairport",
    "airportinternationalkualanamumedan",
    "airportkualanamuinternationalairport",
    "kualanamuairportinternasional",
    "kualanamuinternasionalairportmedan",
    "kualanamuinternationl",
    "airportkualanamumedan",
    "internationalairportkualanamu",
    "kualanamuiternationalairportmedan",
    "airportkualanamumedanindonesia",
    "kualamanuinternationalairportmedan",
]
train["name"].loc[idx] = (
    train["name"].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
)
l1 = [
    "ngurahraiinternationalairport",
    "ngurahraiairport",
    "igustingurahraiinternationalairport",
    "dpsngurahraiinternationalairport",
    "airportngurahraidenpasarbali",
    "ngurahraiinternationalairportairport",
    "airportngurahraiinternationalairport",
    "ngurahraiinternatioanlairportbali",
]
train["name"].loc[idx] = (
    train["name"].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
)
l1 = [
    "adisuciptointernationalairport",
    "airportadisucipto",
    "adisuciptointernasionalairport",
    "airportadisuciptoyogyakarta",
    "adisutjiptoairportjogjakarta",
    "airportadisutjipto",
]
train["name"].loc[idx] = (
    train["name"].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
)
# many names in one nested list
ll1 = [
    [
        "starbucks",
        "starbucksstarbucks",
        "staarbakhs",
        "starbuck",
        "starbucksstaarbakhs",
        "sturbucks",
        "starbuckscoffe",
        "starbaks",
        "starbacks",
        "starbuks",
        "starbucksdrivethru",
    ],
    ["walmart", "wallmart", "wallyworld"],
    [
        "mcdonalds",
        "mcdonaldss",
        "macdonalds",
        "makdonals",
        "macdonals",
        "mcdonals",
        "macdonald",
        "mccafe",
        "mcdonaldssmccafe",
        "mcd",
        "mcdonaldssdrivethru",
        "mcdrive",
        "mcdonaldssmaidanglao",
        "mcdonaldssmkdwnldz",
        "mcdonaldssdrive",
        "makdak",
    ],
    ["paylessshoesource", "payless", "paylessshoes"],
    ["savealot", "savelot"],
    ["subway", "subwaysbwy", "subwayrestaurants"],
    ["burgerking", "burguerking"],
    ["pizzahut", "pizzahutdelivery", "pizzahutexpress"],
    ["dunkin", "dunkindnkndwnts"],
    ["firestone", "firestoneautocare"],
    ["dominos", "dominopizza"],
    [
        "uspostoffice",
        "postoffice",
        "unitedstatespostalservice",
        "usps",
        "unitedstatespostoffice",
    ],
    [
        "jcpenney",
        "jcpenny",
        "jcpopticalpopupshop",
        "jcpenneyoptical",
        "jcpenneysalon",
        "jcpennysalon",
        "jcpenneyportraitstudio",
    ],
    ["enterpriserentacar", "enterprise", "enterprisecarvanhire"],
    ["littlecaesarspizza", "littlecaesars", "littleceasars"],
    ["nettomarkendiscount", "nettofiliale", "netto"],
    ["kroger", "krogerfuel", "krogerpharmacy", "krogermarketplace"],
    ["tgifridays", "fridays"],
    ["ngurahraiinternationalairport", "airportngurahrai"],
    ["tangkubanperahu", "tangkubanparahu"],
    [
        "adisuciptointernationalairport",
        "adisuciptoairport",
        "adisutjiptoairport",
        "adisutjiptointernationalairport",
    ],
    ["kualanamuinternationalairport", "kualanamuinternasionalairport"],
    ["esenlerotogar", "esenlerotogari"],
    ["hydepark", "hydeparklondon", "londonhydepark", "hidepark", "haydpark"],
    ["tesco", "tescoexpress"],
    ["gatwickairport", "londongatwickairport", "gatwick", "englandgatwickairport"],
    [
        "heathrowairport",
        "londonheathrowairport",
        "heathrowairportlondon",
        "heathrow",
        "londonheathrowinternationalairport",
        "heatrowairport",
        "londonheathrowairportairport",
        "londonheathrowairportengland",
        "heathrowunitedkingdom",
        "heatrow",
        "lhrairport",
    ],
    [
        "metroadmiralteiskaia",
        "admiralteiskaia",
        "stantsiiametroadmiralteiskaia",
        "admiralteiskaiametro",
        "metroadmiralteiskaiametroadmiralteyskaya",
        "admiralteiskaiastmetro",
    ],
    ["kfc", "kfccoffee", "kfckfc", "kfcdrivethru"],
    ["cvspharmacy", "cvs"],
    ["chasebank", "chase"],
    ["tgifridays", "tgifriday", "tgif"],
    ["costacoffee", "costa"],
    ["jcpenney", "jcpennys"],
    ["santanderbank", "santander", "bancosantander"],
    ["coffeebean", "coffeebeantealeaf", "coffebeantealeaf"],
    [
        "antalyaairport",
        "antalyainternationalairport",
        "antalyaairportairport",
        "antalyaairportayt",
        "antalyadishatlarairporti",
    ],
    [
        "esenbogaairport",
        "ankaraesenbogaairport",
        "esenbogaairportairporti",
        "esenbogaairportairport",
        "esenbogaairportankara",
        "esenbogainternationalairport",
        "ankaraesenbogainternationalairport",
    ],
    [
        "sabihagokcenairport",
        "istanbulsabihagokcenairport",
        "sabihagokcen",
        "sabihagokceninternationalairport",
        "sabihagokcenuluslararasiairport",
        "istanbulsabihagokcenuluslararasiairport",
        "sabihagokcenairportdishatlar",
        "sabihagokcendishatlar",
        "istanbulsabihagokcen",
        "istanbulsabihagokceninternationalairport",
        "sabihagokceninternetionalairport",
    ],
]
for l1 in ll1:
    train["name"] = train["name"].apply(lambda x: x if x not in l1[1:] else l1[0])


def replace_common_words(s):
    for x in ["kebap", "kebab", "kepab", "kepap"]:
        s = s.replace(x, "kebab")
    for x in ["aloonaloon"]:  # center place in indonesian villages
        s = s.replace(x, "alunalun")
    for x in ["restoram"]:
        s = s.replace(x, "restaurant")
    s = s.replace("internationalairport", "airport")
    return s


train["name"] = train["name"].apply(replace_common_words)


# define cat2 (clean category with low cardinality)
# base it on address, name and catogories - after those have been cleaned (then do not need to include misspellings)
train["cat2"] = ""  # init (left: 129824*)
all_words = (
    {  # map all words in 'values' to 'keys'. Apply to address, name and categories
        "store": [
            "dollartree",
            "circlek",
            "apteka",
            "bricomarche",
            "verizon",
            "relay",
            "firestone",
            "alfamart",
            "walgreens",
            "carrefour",
            "gamestop",
            "radioshack",
            "ikea",
            "walmart",
            "7eleven",
            "bodega",
            "market",
            "boutiqu",
            "store",
            "supermarket",
            "shop",
            "grocer",
            "pharmac",
        ],
        "restaurant": [
            "warunkupnormal",
            "mado",
            "dominos",
            "solaria",
            "bistro",
            "food",
            "shawarma",
            "tearoom",
            "meatball",
            "soup",
            "breakfast",
            "bbq",
            "sushi",
            "ramen",
            "noodle",
            "burger",
            "sandwich",
            "cafe",
            "donut",
            "restaurant",
            "coffeeshop",
            "buffet",
            "pizzaplace",
            "diner",
            "steakhouse",
            "kitchen",
            "foodcourt",
            "baker",
            "starbucks",
            "dunkin",
            "tacoplac",
            "snackplac",
        ],
        "fastfood": [
            "teremok",
            "chickfil",
            "arbys",
            "popeyes",
            "chilis",
            "dairyqueen",
            "tacobell",
            "wendys",
            "burgerking",
            "fastfood",
            "kfc",
            "subway",
            "pizzahut",
            "mcdonalds",
            "friedchicken",
        ],
        "school": [
            "sororityhous",
            "fraternity",
            "college",
            "school",
            "universit",
            "classroom",
            "student",
        ],
        "housing": ["home", "housing", "residential", "building", "apartment", "condo"],
        "bank": ["creditunion", "bank", "atm"],
        "airport": [
            "baggageclaim",
            "airport",
            "terminal",
            "airline",
            "baggagereclaim",
            "concourse",
        ],
        "venue": [
            "photographystudio",
            "bowlingalle",
            "cineplex",
            "cinema",
            "ballroom",
            "stadium",
            "meetingroom",
            "conference",
            "convention",
            "entertainment",
            "venue",
            "auditorium",
            "multiplex",
            "eventspace",
            "opera",
            "concert",
            "theater",
            "megaplex",
        ],
        "museum": ["museum", "galler"],
        "church": [
            "sacred",
            "shrine",
            "spiritual",
            "mosque",
            "temple",
            "cathedral",
            "church",
            "christ",
        ],
        "park": ["zoo", "park"],
        "bar": [
            "sportclips",
            "speakeas",
            "buffalowildwing",
            "brewer",
            "pub",
            "bar",
            "nightclub",
            "nightlife",
            "lounge",
        ],
        "station": [
            "flyingj",
            "pilottravel",
            "shell",
            "bikerent",
            "rail",
            "station",
            "train",
            "metro",
            "bus",
            "stantsiia",
        ],
        "medical": [
            "poliklinika",
            "diagnos",
            "veterinarian",
            "emergencyroom",
            "hospital",
            "medical",
            "doctor",
            "dentist",
        ],
        "gym": ["clinic", "wellnes", "sportsclub", "gym", "fitnes", "athletic"],
        "outdoor": [
            "farm",
            "bagevi",
            "bridges",
            "surf",
            "dogrun",
            "sceniclookout",
            "campground",
            "golfcours",
            "forest",
            "river",
            "outdoor",
            "beach",
            "field",
            "plaza",
            "lake",
            "playground",
            "mountain",
            "pool",
            "basketballcourt",
            "garden",
        ],
        "office": [
            "techstartup",
            "hrblock",
            "work",
            "creditrepair",
            "librari",
            "coworkingspaces",
            "office",
            "service",
            "lawyer",
            "courthous",
            "cityhall",
            "notarius",
        ],
        "carrental": [
            "rentalcar",
            "hertz",
            "rentacar",
            "aviscarrent",
            "dollarrent",
            "zipcar",
            "autodealership",
            "carwashes",
        ],
        "hotel": ["boardinghous", "hostel", "hotel", "motel"],
    }
)

for col in ["address", "categories", "name"]:
    for word in all_words.keys():
        words = all_words[word]
        for w in words:
            train["cat2"].loc[train[col].str.contains(w, regex=False)] = word

cat2_di = {
    "": 0,
    "restaurant": 1,
    "bar": 2,
    "store": 3,
    "housing": 4,
    "office": 5,
    "outdoor": 6,
    "station": 7,
    "medical": 8,
    "venue": 9,
    "hotel": 10,
    "school": 11,
    "church": 12,
    "park": 13,
    "bank": 14,
    "airport": 15,
    "gym": 16,
    "museum": 17,
    "carrental": 18,
    "fastfood": 19,
}
train["cat2"] = train["cat2"].map(cat2_di).astype("int16")
print("finished cat2", int(time.time() - start_time), "sec")


# get average dist between true matches by cat2/category_simpl - hardcode it for submission
##pairs = pd.read_csv('input/pairs.csv')
##pairs = pairs.merge(train[['id','category_simpl']], left_on='id_1', right_on='id', how='left')
##pairs = pairs.merge(train[['id','category_simpl']], left_on='id_2', right_on='id', how='left')
##pairs['dist']=  distance(np.array(pairs['latitude_1']), np.array(pairs['longitude_1']), np.array(pairs['latitude_2']), np.array(pairs['longitude_2']))
##pairs['cm'] = np.maximum(0, pairs['category_simpl_x'])
##pairs['cm'].iloc[pairs['category_simpl_x'] != pairs['category_simpl_y']] = 0 # not a match - call it 0
##a=pairs.loc[pairs['match'] == True].groupby('cm')['dist'].agg(['median','size']).reset_index()
##ll = list(np.round(a['median'],0).astype(np.int32))
##stop


# median by cat2
dist_by_cat2 = {
    0: 176,
    1: 61,
    2: 67,
    3: 95,
    4: 97,
    5: 90,
    6: 270,
    7: 126,
    8: 111,
    9: 121,
    10: 75,
    11: 92,
    12: 122,
    13: 205,
    14: 78,
    15: 1112,
    16: 85,
    17: 98,
    18: 59,
    19: 133,
}
# median by category_simpl
dist_by_category_simpl = {
    0: 135,
    1: 1108,
    2: 80,
    3: 89,
    4: 280,
    5: 52,
    6: 78,
    7: 79,
    8: 81,
    9: 153,
    10: 97,
    11: 165,
    12: 67,
    13: 95,
    14: 80,
    15: 107,
    16: 679,
    17: 65,
    18: 149,
    19: 39,
    20: 90,
    21: 100,
    22: 84,
    23: 119,
    24: 81,
    25: 84,
    26: 80,
    27: 76,
    28: 56,
    29: 143,
    30: 227,
    31: 132,
    32: 92,
    33: 31,
    34: 147,
    35: 146,
    36: 65,
    37: 70,
    38: 312,
    39: 75,
    40: 83,
    41: 64,
    42: 50,
    43: 131,
    44: 57,
    45: 71,
    46: 113,
    47: 448,
    48: 1409,
    49: 30,
    50: 273,
    51: 127,
    52: 145,
    53: 97,
    54: 68,
    55: 83,
    56: 37,
    57: 136,
    58: 5351,
    59: 97,
    60: 97,
    61: 739,
    62: 31,
    63: 66,
    64: 83,
    65: 78,
    66: 79,
    67: 87,
    68: 87,
    69: 231,
    70: 129,
    71: 612,
    72: 131,
    73: 144,
    74: 85,
    75: 111,
    76: 37,
    77: 72,
    78: 93,
    79: 361,
    80: 234,
    81: 67,
    82: 408,
    83: 109,
    84: 77,
    85: 44,
    86: 76,
    87: 88,
    88: 30,
    89: 136,
    90: 302,
    91: 152,
    92: 66,
    93: 52,
}


# feature: count of distinct substrings of length >=X in both names
def cc_lcs(str1, str2, x):
    c = 0  # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(
            0, len(str1), 0, len(str2)
        )
        if d.size < x:  # no more X+ substrings - exit
            return c
        c += 1
        # remove found substring
        if d.a > 0:
            str1 = str1[: d.a] + str1[d.a + d.size :]
        else:
            str1 = str1[d.size :]
        if d.b > 0:
            str2 = str2[: d.b] + str2[d.b + d.size :]
        else:
            str2 = str2[d.size :]


# feature: total length of distinct substrings of length >=X in both names
def ll_lcs(str1, str2, x):
    c = 0  # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(
            0, len(str1), 0, len(str2)
        )
        if d.size < x:  # no more X+ substrings - exit
            return c
        c += d.size
        # remove found substring
        if d.a > 0:
            str1 = str1[: d.a] + str1[d.a + d.size :]
        else:
            str1 = str1[d.size :]
        if d.b > 0:
            str2 = str2[: d.b] + str2[d.b + d.size :]
        else:
            str2 = str2[d.size :]


# include all col - regardless of the shift********************************************
# phone
p3 = train[["country", "id", "point_of_interest", "phone", "lon2"]].copy()
p3 = (
    p3.loc[p3["phone"] != ""]
    .sort_values(by=["country", "phone", "lon2", "id"])
    .reset_index(drop=True)
)
idx1 = []
idx2 = []
d = p3["phone"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 11):  # 11 = add 10 sets
        if i + j < p3.shape[0] and lcs(d[i], d[i + j]) >= 7:  # accept <=3 digits off
            idx1.append(i)
            idx2.append(i + j)
p1 = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2 = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "phone match: added",
    p1.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# lat/lon, rounded to 2 x 4 digits = 22* meters square; there should not be too many false positives this close to each other
# do this in 4 blocks, shifted by 1/2 size, to avoid cut-offs
for s1 in [0, 5e-5]:
    for s2 in [0, 5e-5]:
        p3 = train[
            ["country", "id", "point_of_interest", "latitude", "longitude"]
        ].copy()
        p3["latitude"] = np.round(s1 + 0.5 * p3["latitude"], 4)  # rounded to 4 digits
        p3["longitude"] = np.round(
            s2 + 0.5 * p3["longitude"] / np.cos(p3["latitude"] * 3.14 / 180.0), 4
        )  # rounded to 4 digits
        p3 = p3.sort_values(by=["country", "latitude", "longitude", "id"]).reset_index(
            drop=True
        )
        idx1 = []
        idx2 = []
        lat, lon = p3["latitude"].to_numpy(), p3["longitude"].to_numpy()
        for i in range(p3.shape[0] - 1):
            for j in range(1, 5):  # 5 = add 4 sets
                if (
                    i + j < p3.shape[0]
                    and lat[i] == lat[i + j]
                    and lon[i] == lon[i + j]
                ):
                    idx1.append(i)
                    idx2.append(i + j)
        p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
        p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
        # append
        p1 = p1.append(p1a, ignore_index=True)
        p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)

print_infos(p1, p2)

a = p1.groupby("id")["y"].sum().reset_index()
print(
    "lat/lon match: added",
    p1.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# url
p3 = train[["country", "id", "point_of_interest", "url", "lon2", "lat2"]].copy()
p3 = (
    p3.loc[p3["url"] != ""]
    .sort_values(by=["country", "url", "lon2", "lat2", "id"])
    .reset_index(drop=True)
)
idx1 = []
idx2 = []
d = p3["url"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 2):  # 2 = add 1 set
        if i + j < p3.shape[0] and ll_lcs(d[i], d[i + j], 3) >= 7:  # ll_lcs(3) >= 7
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "url match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# categories
p3 = train[["country", "id", "point_of_interest", "categories", "lon2", "lat2"]].copy()
p3 = (
    p3.loc[p3["categories"] != ""]
    .sort_values(by=["country", "categories", "lon2", "lat2", "id"])
    .reset_index(drop=True)
)
idx1 = []
idx2 = []
d = p3["categories"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 2):  # 2 = add 1 set
        if i + j < p3.shape[0] and d[i][:4] == d[i + j][:4]:  # match on first 4 leters
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "categories match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# address
p3 = train[["country", "id", "point_of_interest", "address", "lon2", "lat2"]].copy()
p3 = (
    p3.loc[p3["address"] != ""]
    .sort_values(by=["country", "address", "lon2", "lat2", "id"])
    .reset_index(drop=True)
)
idx1 = []
idx2 = []
d = p3["address"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 7):  # 7 = add 6 sets
        if i + j < p3.shape[0] and lcs2(d[i], d[i + j]) >= 6:  # lcs2 >= 6
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "address match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# name
p3 = train[["country", "id", "point_of_interest", "name", "lon2", "lat2"]].copy()
p3 = (
    p3.loc[p3["name"] != ""]
    .sort_values(by=["country", "name", "lon2", "lat2", "id"])
    .reset_index(drop=True)
)
idx1 = []
idx2 = []
d = p3["name"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 4):  # 4 = add 3 sets
        if i + j < p3.shape[0] and lcs2(d[i], d[i + j]) >= 5:  # lcs2 >= 5
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "name match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# latitude
p3 = train[["country", "id", "point_of_interest", "name", "latitude"]].copy()
p3 = p3.sort_values(by=["country", "latitude", "id"]).reset_index(drop=True)
idx1 = []
idx2 = []
d = p3["latitude"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 21):  # 21 = add 20 sets
        if i + j < p3.shape[0] and d[i] == d[i + j]:  # exact match
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "lat match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# longitude
p3 = train[["country", "id", "point_of_interest", "name", "longitude"]].copy()
p3 = p3.sort_values(by=["country", "longitude", "id"]).reset_index(drop=True)
idx1 = []
idx2 = []
d = p3["longitude"].to_numpy()
for i in range(p3.shape[0] - 1):
    for j in range(1, 21):  # 21 = add 20 sets
        if i + j < p3.shape[0] and d[i] == d[i + j]:  # exact match
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "lon match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)

# name1
p3 = train[
    [
        "country",
        "id",
        "point_of_interest",
        "name",
        "latitude",
        "longitude",
        "categories",
    ]
].copy()
# rounded coordinates
p3["latitude"] = np.round(p3["latitude"], 1).astype(
    "float32"
)  # rounding: 1=10Km, 2=1Km
p3["longitude"] = np.round(p3["longitude"], 1).astype("float32")
p3 = p3.sort_values(
    by=["country", "latitude", "longitude", "categories", "id"]
).reset_index(drop=True)
idx1 = []
idx2 = []
names = p3["name"].to_numpy()
lon2 = p3["longitude"].to_numpy()
for i in range(p3.shape[0] - 1):
    if i % 100000 == 0:
        print(i, int(time.time() - start_time), "sec")
    li = lon2[i]
    for j in range(
        1, min(300, p3.shape[0] - 1 - i)
    ):  # put a limit here - look at no more than X items
        if (
            li != lon2[i + j]
        ):  # if lon matches, lat and country also match - b/c of sorting order
            break
        if lcs2(names[i], names[i + j]) >= 5:  # lcs2 >= 5
            idx1.append(i)
            idx2.append(i + j)
p1a = p3[["id", "point_of_interest"]].loc[idx1].reset_index(drop=True)
p2a = p3[["id", "point_of_interest"]].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "name1 match: added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
    "records",
    int(time.time() - start_time),
    "sec",
)
del d, names, lon2, p3, idx1, idx2, p1a, p2a, lat, lon
gc.collect()


# remove duplicate pairs
p12 = pd.concat([p1["id"], p2["id"]], axis=1)
p12.columns = ["id", "id2"]
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12["id"] > p12["id2"]
p12["t"] = p12["id"]
p12["id"].loc[idx] = p12["id2"].loc[idx]
p12["id2"].loc[idx] = p12["t"].loc[idx]

p12 = p12.sort_values(by=["id", "id2"]).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=["id", "id2"])

# also drop id == id2 - it may happen
p12 = p12.loc[p12["id"] != p12["id2"]]
p1 = p1.loc[p12["index"]].reset_index(drop=True)
p2 = p2.loc[p12["index"]].reset_index(drop=True)
del p12, idx
gc.collect()
y = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
print(
    "removed duplicates",
    p1.shape[0],
    (p1["point_of_interest"] == p2["point_of_interest"]).sum(),
    int(time.time() - start_time),
    "sec",
)
# get stats
p1["y"] = y
a = p1.groupby("id")["y"].sum().reset_index()
print("count of all matching pairs is:", np.minimum(1, a["y"]).sum())


def substring_ratio(name1, name2):
    N = (len(name1) + len(name2)) / 2
    if N == 0:
        return 0
    substr = lcs2(name1, name2)
    return substr / N


def subseq_ratio(name1, name2):
    N = (len(name1) + len(name2)) / 2
    if N == 0:
        return 0
    substr = lcs(name1, name2)
    return substr / N


print_infos(p1, p2)

# sort to put similar points next to each other - for constructing pairs
sort = [
    "lat2",
    "lon2",
    "name2",
    "latitude",
    "city",
    "cat2",
    "name",
    "address",
    "country",
    "id",
]
train = train.sort_values(by=sort).reset_index(drop=True)
train.drop(
    ["lat2", "lon2", "name2"], axis=1, inplace=True
)  # these are no longer needed
print("finished sorting", int(time.time() - start_time), "sec")

# construct pairs***********************************************************************************************************
cols = ["id", "latitude", "longitude", "point_of_interest", "name", "category_simpl"]
colsa = ["id", "point_of_interest"]
p1a = train[colsa].copy()
p2a = train[colsa].iloc[1:, :].reset_index(drop=True).copy()
p2a = p2a.append(train[colsa].iloc[0], ignore_index=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
# add more shifts, only for short distances or for partial name matches
for i, s in enumerate(range(2, 121)):  # 121
    if s == 15:  # resort by closer location after 15 shifts
        train["lat2"] = np.round(train["latitude"], 2).astype("float32")  # 2 = 1 Km
        train["lon2"] = np.round(train["longitude"], 2).astype("float32")
        train = train.sort_values(
            by=["country", "lat2", "lon2", "categories", "city", "id"]
        ).reset_index(drop=True)
        train.drop(["lat2", "lon2"], axis=1, inplace=True)
    if s < 4:
        maxdist = 500000
    elif s < 8:
        maxdist = 10000
    elif s < 12:
        maxdist = 5000
    elif s < 15:
        maxdist = 2000
    else:
        maxdist = max(100, 200 - (s - 16) * 1)
    s2 = s  # shift
    if i >= 13:  # resorted data
        s2 = i - 12
    p2a = train[cols].iloc[s2:, :]
    p2a = p2a.append(train[cols].iloc[:s2, :], ignore_index=True)

    # drop pairs with large distances
    dist = distance(
        np.array(train["latitude"]),
        np.array(train["longitude"]),
        np.array(p2a["latitude"]),
        np.array(p2a["longitude"]),
    )
    same_cat_simpl = (train["category_simpl"] == p2a["category_simpl"]) & (
        train["category_simpl"] > 0
    )

    ii = np.zeros(train.shape[0], dtype=np.int8)
    x1, x2 = train["name"].to_numpy(), p2a["name"].to_numpy()
    for j in range(train.shape[0]):
        if pi1(x1[j], x2[j]):
            ii[j] = 1
        elif substring_ratio(x1[j], x2[j]) >= 0.65:
            ii[j] = 1
        elif subseq_ratio(x1[j], x2[j]) >= 0.75:
            ii[j] = 1
        elif len(x1[j]) >= 7 and len(x2[j]) >= 7 and x1[j].endswith(x2[j][-7:]):
            ii[j] = 1
    # keep if dist < maxdist, or names partially match
    # idx = (dist < maxdist) | (ii > 0)
    idx = (
        (dist < maxdist)
        | (ii > 0)
        | np.logical_and(same_cat_simpl, dist < train["q90"] * 900)
    )

    p1 = p1.append(train[colsa].loc[idx], ignore_index=True)
    p2 = p2.append(p2a[colsa].loc[idx], ignore_index=True)
    if s % 10 == 0:
        # get stats; overstated b/c dups are not excluded yet
        p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(
            np.int8
        )
        print_infos(p1, p2)
        a = p1.groupby("id")["y"].sum().reset_index()
        print(
            i,
            maxdist,
            s,
            s2,
            p1.shape[0],
            p1["y"].sum(),
            np.minimum(1, a["y"]).sum(),
            int(time.time() - start_time),
            "sec",
        )
    gc.collect()
del p1a, p2a, dist, idx
gc.collect()


# remove duplicate pairs
p12 = pd.concat([p1["id"], p2["id"]], axis=1)
p12.columns = ["id", "id2"]
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12["id"] > p12["id2"]
p12["t"] = p12["id"]
p12["id"].loc[idx] = p12["id2"].loc[idx]
p12["id2"].loc[idx] = p12["t"].loc[idx]

p12 = p12.sort_values(by=["id", "id2"]).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=["id", "id2"])

# also drop id == id2 - it may happen
p12 = p12.loc[p12["id"] != p12["id2"]]
p1 = p1.loc[p12["index"]].reset_index(drop=True)
p2 = p2.loc[p12["index"]].reset_index(drop=True)
del p12, idx
gc.collect()
y = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
print(
    "removed duplicates",
    p1.shape[0],
    (p1["point_of_interest"] == p2["point_of_interest"]).sum(),
    int(time.time() - start_time),
    "sec",
)
# get stats
p1["y"] = y
a = p1.groupby("id")["y"].sum().reset_index()
print("count of all matching pairs is:", np.minimum(1, a["y"]).sum())


# Add close candidates
from scipy import spatial


def Compute_Mdist_Mindex(matrix, nb_max_candidates=1000, thr_distance=None):

    nb_max_candidates = min(nb_max_candidates, len(matrix))
    Z = tuple(zip(matrix[:, 0], matrix[:, 1]))  # latitude, longitude

    tree = spatial.KDTree(Z)
    M_dist, M_index = tree.query(Z, min(nb_max_candidates, len(matrix)))

    if thr_distance is None:
        return M_dist, M_index

    # Threshold filter
    Nb_matches_potentiels = []
    for i in range(len(M_index)):
        n = len([d for d in M_dist[i] if d <= thr_distance])
        Nb_matches_potentiels.append(n)
    M_dist = [m[: Nb_matches_potentiels[i]] for i, m in enumerate(M_dist)]
    M_index = [m[: Nb_matches_potentiels[i]] for i, m in enumerate(M_index)]

    return M_dist, M_index


New_candidates = []

# for country_ in range(2, len(countries)+1):
for country_ in [2, 3]:

    nb_max_candidates = 400
    new_cand = set()

    # Create matrix
    matrix = train[train["country"] == country_].copy()
    if len(matrix) <= 1:
        break
    Original_idx = {i: idx for i, idx in enumerate(matrix.index)}

    # Find closest neighbours
    M_dist, M_index = Compute_Mdist_Mindex(
        matrix[["latitude", "longitude"]].to_numpy(),
        nb_max_candidates=nb_max_candidates,
    )

    # Select candidates
    new_true_match = 0
    infos = matrix[["id", "name", "point_of_interest"]].to_numpy()

    for idx1, (Liste_idx, Liste_val) in enumerate(zip(M_index, M_dist)):
        for idx2, dist in zip(Liste_idx, Liste_val):
            if idx1 == idx2:
                continue

            # Too far candidates
            if dist > 0.12:
                break

            id1, id2 = infos[idx1, 0], infos[idx2, 0]
            name1, name2 = infos[idx1, 1], infos[idx2, 1]

            if pi1(name1, name2) == 1 or substring_ratio(name1, name2) >= 0.5:
                key = tuple(sorted([id1, id2]))

    # Add new candidates
    New_candidates += [list(x) for x in new_cand]
    print(
        f"Country {country_} ({countries[country_-1]}) : {new_true_match}/{len(new_cand)} new cand added."
    )

# Add matches
size1 = len(p1)
Added_p1, Added_p2 = [], []
for idx1, idx2 in New_candidates:
    id1, id2 = train["id"].iat[idx1], train["id"].iat[idx2]
    poi1, poi2 = (
        train["point_of_interest"].iat[idx1],
        train["point_of_interest"].iat[idx2],
    )
    Added_p1.append([id1, poi1, 0])
    Added_p2.append([id2, poi2])
Added_p1 = pd.DataFrame(Added_p1, columns=p1.columns)
Added_p2 = pd.DataFrame(Added_p2, columns=p2.columns)
for col in Added_p1.columns:
    Added_p1[col] = Added_p1[col].astype(p1[col].dtype)
for col in Added_p2.columns:
    Added_p2[col] = Added_p2[col].astype(p2[col].dtype)
p1 = p1.append(Added_p1).reset_index(drop=True).copy()
p2 = p2.append(Added_p2).reset_index(drop=True).copy()
print(f"Candidates added : {len(p1)-size1}/{len(p1)}.")


# remove duplicate pairs
p12 = pd.concat([p1["id"], p2["id"]], axis=1)
p12.columns = ["id", "id2"]
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12["id"] > p12["id2"]
p12["t"] = p12["id"]
p12["id"].loc[idx] = p12["id2"].loc[idx]
p12["id2"].loc[idx] = p12["t"].loc[idx]

p12 = p12.sort_values(by=["id", "id2"]).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=["id", "id2"])

# also drop id == id2 - it may happen
p12 = p12.loc[p12["id"] != p12["id2"]]
p1 = p1.loc[p12["index"]].reset_index(drop=True)
p2 = p2.loc[p12["index"]].reset_index(drop=True)
del p12, idx, matrix
gc.collect()
y = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
print(
    "removed duplicates",
    p1.shape[0],
    (p1["point_of_interest"] == p2["point_of_interest"]).sum(),
    int(time.time() - start_time),
    "sec",
)
# get stats
p1["y"] = y
a = p1.groupby("id")["y"].sum().reset_index()
print("count of all matching pairs is:", np.minimum(1, a["y"]).sum())


# candidates in initial Youri's solution
ID_to_POI = dict(zip(train["id"], train["point_of_interest"]))
nb_true_matchs_initial = 0
Cand = {}
for i, (id1, id2) in enumerate(zip(p1["id"], p2["id"])):
    key = f"{min(id1, id2)}-{max(id1, id2)}"
    Cand[key] = p1["y"].iloc[i]
    nb_true_matchs_initial += int(ID_to_POI[id1] == ID_to_POI[id2])


#################################################################################
## TF-IDF n°1 : airports
#################################################################################
# Vincent

far_cat_simpl = [1, 2]
thr_tfidf = 0.45

for col_name in ["name_initial_decode"]:

    Names = train[train["category_simpl"].isin(far_cat_simpl + [-1])][
        col_name
    ].copy()  # add unknown categories

    def process_terminal(text):
        for i in range(0, 30):
            text = text.replace(f"terminal {i}", "")
            text = text.replace(f"terminal{i}", "")
            text = text.replace(f"t{i}", "")
        return text

    Names = Names.apply(process_terminal)

    # Drop stop words
    Names = Names.apply(lambda x: x.replace("airpord", "airport"))
    Names = Names.apply(lambda x: x.replace("internasional", "international"))
    Names = Names.apply(lambda x: x.replace("internacional", "international"))
    for stopword in [
        "terminal",
        "airport",
        "arrival",
        "hall",
        "departure",
        "bus stop",
        "airways",
        "checkin",
    ]:
        Names = Names.apply(lambda x: x.replace(stopword + "s", ""))
        Names = Names.apply(lambda x: x.replace(stopword, ""))
    Names = Names.apply(lambda x: x.strip())
    Names = Names[Names.str.len() >= 2]

    Names_numrow = {
        i: idx for i, idx in enumerate(Names.index)
    }  # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=thr_tfidf)

        # no self-matchs and retrieve the initial row number
        Tfidf_no_selfmatch = [
            [Names_numrow[i], [Names_numrow[x] for x in L if x != i]]
            for i, L in enumerate(Tfidf_idx)
        ]
        Tfidf_no_selfmatch = [x for x in Tfidf_no_selfmatch if len(x[-1]) > 0]
        print("Nb cand tf-idf :", sum([len(L) for idx, L in Tfidf_no_selfmatch]))

        # Add matches
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, Liste_idx in Tfidf_no_selfmatch:
            id1, name1, lat1, lon1 = (
                train["id"].iat[idx1],
                train["name"].iat[idx1],
                train["latitude"].iat[idx1],
                train["longitude"].iat[idx1],
            )
            cat1, country1, cat_simpl1 = (
                train["categories"].iat[idx1],
                train["country"].iat[idx1],
                train["category_simpl"].iat[idx1],
            )
            for idx2 in Liste_idx:
                # if len(Liste_idx)>30 : continue
                if idx1 < idx2:
                    id2, lat2, lon2 = (
                        train["id"].iat[idx2],
                        train["latitude"].iat[idx2],
                        train["longitude"].iat[idx2],
                    )
                    cat2, country2, cat_simpl2 = (
                        train["categories"].iat[idx2],
                        train["country"].iat[idx2],
                        train["category_simpl"].iat[idx2],
                    )
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    # same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='')
                    if (
                        key not in Cand
                        and (cat_simpl1 == 1 or cat_simpl2 == 1)
                        and (
                            haversine(lat1, lon1, lat2, lon2) <= 100
                            or "kualalumpur" in name1
                        )
                    ):
                        poi1, poi2 = (
                            train["point_of_interest"].iat[idx1],
                            train["point_of_interest"].iat[idx2],
                        )
                        Cand[key] = int(poi1 == poi2)
                        Added_p1.append([id1, poi1, int(poi1 == poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added for tfidf n°1 (airports) : {len(p1)-size1}/{len(p1)}.")


#################################################################################
## TF-IDF n°2 : metro stations
#################################################################################
# Vincent

far_cat_simpl = [4]
thr_tfidf = 0.45
thr_distance = 100

for col_name in ["name_initial", "name_initial_decode"]:

    Names = train[train["category_simpl"].isin(far_cat_simpl)][
        col_name
    ].copy()  # add unknown categories

    # Drop stop words
    for stopword in ["stasiun", "station", "metro", "北改札", "bei gai zha", "stasiun"]:
        Names = Names.apply(lambda x: x.replace(stopword + "s", ""))
        Names = Names.apply(lambda x: x.replace(stopword, ""))
    Names = Names.apply(lambda x: x.strip())
    Names = Names[Names.str.len() > 2]

    Names_numrow = {
        i: idx for i, idx in enumerate(Names.index)
    }  # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=thr_tfidf)

        # no self-matchs and retrieve the initial row number
        Tfidf_no_selfmatch = [
            [Names_numrow[i], [Names_numrow[x] for x in L if x != i]]
            for i, L in enumerate(Tfidf_idx)
        ]
        Tfidf_no_selfmatch = [x for x in Tfidf_no_selfmatch if len(x[-1]) > 0]
        print("Nb cand tf-idf :", sum([len(L) for idx, L in Tfidf_no_selfmatch]))

        # Add matches
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, Liste_idx in Tfidf_no_selfmatch:
            id1, lat1, lon1, cat1, cat_simpl1 = (
                train["id"].iat[idx1],
                train["latitude"].iat[idx1],
                train["longitude"].iat[idx1],
                train["categories"].iat[idx1],
                train["category_simpl"].iat[idx1],
            )
            for idx2 in Liste_idx:
                if idx1 < idx2:
                    id2, lat2, lon2, cat2, cat_simpl2 = (
                        train["id"].iat[idx2],
                        train["latitude"].iat[idx2],
                        train["longitude"].iat[idx2],
                        train["categories"].iat[idx2],
                        train["category_simpl"].iat[idx2],
                    )
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    if (
                        key not in Cand
                        and haversine(lat1, lon1, lat2, lon2) <= thr_distance
                        and (cat_simpl1 in far_cat_simpl or cat_simpl2 in far_cat_simpl)
                    ):
                        poi1, poi2 = (
                            train["point_of_interest"].iat[idx1],
                            train["point_of_interest"].iat[idx2],
                        )
                        Cand[key] = int(poi1 == poi2)
                        Added_p1.append([id1, poi1, int(poi1 == poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns))
            .reset_index(drop=True)
            .copy()
        )
        print(
            f"Candidates added for tfidf n°2 (metro stations) : {len(p1)-size1}/{len(p1)}."
        )


#################################################################################
## TF-IDF n°3a : for each countries (with initial unprocessed name)
#################################################################################
# Vincent

thr_tfidf_ = 0.5
thr_distance_ = 20
thr_distance_or_same_cat_ = 2

size = len(p1)

for country in [2, 3, 32]:  # range(1, 30)

    # Reset parameters
    thr_tfidf = thr_tfidf_
    thr_distance = thr_distance_
    thr_distance_or_same_cat = thr_distance_or_same_cat_

    # Tune parameters for each country
    if country == 2:
        thr_tfidf = 1.1  # will impose to have same category
        thr_distance = 20
        thr_distance_or_same_cat = -1  # will impose to have same category
    elif country == 3:
        thr_tfidf = 0.6
        thr_distance = 10
    elif country == 32:
        thr_tfidf = 0.4
        thr_distance = 100  # no limit
        thr_distance_or_same_cat = 100  # no limit

    # List of names
    Names = train[train["country"] == country]["name_initial"].copy()
    if len(Names) == 0:
        break

    print()
    print("#" * 20)
    print(f"# Country n°{country} : {countries[country-1]}.")

    Names_numrow = {
        i: idx for i, idx in enumerate(Names.index)
    }  # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=min(0.45, thr_tfidf))

        # no self-matchs and retrieve the initial row number
        Tfidf_idx = [[Names_numrow[x] for x in L] for L in Tfidf_idx]

        # Add matches : /!\ ONLY IF THERE IS A CATEGORY MATCH AND THE DISTANCE IS NOT TOO BIG
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, (Liste_idx, Liste_val) in enumerate(zip(Tfidf_idx, Tfidf_val)):
            idx1 = Names_numrow[idx1]
            id1, lat1, lon1, cat1, cat_simpl1 = (
                train["id"].iat[idx1],
                train["latitude"].iat[idx1],
                train["longitude"].iat[idx1],
                train["categories"].iat[idx1],
                train["category_simpl"].iat[idx1],
            )
            for idx2, val in zip(Liste_idx, Liste_val):
                if idx1 < idx2:
                    id2, lat2, lon2, cat2, cat_simpl2 = (
                        train["id"].iat[idx2],
                        train["latitude"].iat[idx2],
                        train["longitude"].iat[idx2],
                        train["categories"].iat[idx2],
                        train["category_simpl"].iat[idx2],
                    )
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    dist = haversine(lat1, lon1, lat2, lon2)
                    same_cat = (cat_simpl1 == cat_simpl2 and cat_simpl1 > 0) or (
                        cat1 == cat2 and cat1 != "" and cat1 != "nan"
                    )
                    if (
                        key not in Cand
                        and dist <= thr_distance
                        and (same_cat or dist <= thr_distance_or_same_cat)
                        and (same_cat or val >= thr_tfidf)
                    ):
                        poi1, poi2 = (
                            train["point_of_interest"].iat[idx1],
                            train["point_of_interest"].iat[idx2],
                        )
                        Cand[key] = int(poi1 == poi2)
                        Added_p1.append([id1, poi1, int(poi1 == poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added : {len(p1)-size1}/{len(p1)}.")
print("\n-> TF-IDF for contries finished.")
print(f"Candidates added : {len(p1)-size}.")


#################################################################################
## TF-IDF n°3b : for each countries (with few processed name)
#################################################################################
# Vincent

thr_tfidf_ = 0.45
thr_distance_ = 25
thr_distance_or_same_cat_ = 10

size = len(p1)

for country in [32]:  # range(1, 30)

    # Reset parameter
    thr_tfidf = thr_tfidf_
    thr_distance = thr_distance_
    thr_distance_or_same_cat = thr_distance_or_same_cat_

    # Tune parameters for each country
    if country == 32:
        thr_tfidf_ = 0.4
        thr_distance = 100  # no limit
        thr_distance_or_same_cat = 100  # no limit

    Names = train[train["country"] == country]["name_initial_decode"].copy()
    if len(Names) == 0:
        break

    print()
    print("#" * 20)
    print(f"# Country n°{country} : {countries[country-1]}.")

    Names_numrow = {
        i: idx for i, idx in enumerate(Names.index)
    }  # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=min(0.4, thr_tfidf))

        # no self-matchs and retrieve the initial row number
        Tfidf_idx = [[Names_numrow[x] for x in L] for L in Tfidf_idx]

        # Add matches : /!\ ONLY IF THERE IS A CATEGORY MATCH AND THE DISTANCE IS NOT TOO BIG
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, (Liste_idx, Liste_val) in enumerate(zip(Tfidf_idx, Tfidf_val)):
            idx1 = Names_numrow[idx1]
            id1, lat1, lon1, cat1, cat_simpl1 = (
                train["id"].iat[idx1],
                train["latitude"].iat[idx1],
                train["longitude"].iat[idx1],
                train["categories"].iat[idx1],
                train["category_simpl"].iat[idx1],
            )
            for idx2, val in zip(Liste_idx, Liste_val):
                if idx1 < idx2:
                    id2, lat2, lon2, cat2, cat_simpl2 = (
                        train["id"].iat[idx2],
                        train["latitude"].iat[idx2],
                        train["longitude"].iat[idx2],
                        train["categories"].iat[idx2],
                        train["category_simpl"].iat[idx2],
                    )
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    dist = haversine(lat1, lon1, lat2, lon2)
                    same_cat = (cat_simpl1 == cat_simpl2 and cat_simpl1 > 0) or (
                        cat1 == cat2 and cat1 != "" and cat1 != "nan"
                    )
                    if (
                        key not in Cand
                        and dist <= thr_distance
                        and (same_cat or dist <= thr_distance_or_same_cat)
                        and (same_cat or val >= thr_tfidf)
                    ):
                        poi1, poi2 = (
                            train["point_of_interest"].iat[idx1],
                            train["point_of_interest"].iat[idx2],
                        )
                        Cand[key] = int(poi1 == poi2)
                        Added_p1.append([id1, poi1, int(poi1 == poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added : {len(p1)-size1}.")
print("\n-> TF-IDF for contries finished.")
print(f"Candidates added : {len(p1)-size}.")


#################################################################################
## Add candidates based on same name/phone/address
#################################################################################
# Vincent


def create_address(address, city):
    if address == "nan" or len(address) <= 1:
        return "nan"
    elif city == "nan" or len(city) <= 1:
        return "nan"
    else:
        return address.lower().strip() + "-" + city.lower().strip()


def find_potential_matchs(row):
    name = row["name"]
    # Missing name
    if (name not in work_names) or name == "nan" or len(name) <= 1:
        return []

    phone = row["phone"]
    address_complet = create_address(row["address"], row["city"])

    # Non-missing name
    index = work_names[name].copy()

    if phone != "nan" and len(phone) > 1 and phone in work_phones:
        index += work_phones[phone]

    if address_complet != "nan" and address_complet in work_address:
        index += work_address[address_complet]

    return list(set(index))
