import difflib
import numpy as np
import pandas as pd
from numba import jit
from sys import getsizeof
from scipy import spatial
from numerize.numerize import numerize
from math import radians, cos, sin, asin, sqrt
from sklearn.feature_extraction.text import TfidfVectorizer

##########################
#           CV           #
##########################


def cci(x, y):  # count y in x (intersection)
    i = 0
    for j in range(1000):
        l1 = y.find(" ")
        if y[: l1 + 1] in x:
            # include space in search to avoid
            # false positives(need to have trailing space in m_true)
            i += 1
        y = y[l1 + 1:]
        if " " not in y:  # only 1 item left
            break
    return i


def get_CV(p1, p2, y, oof_preds, train):
    cv0 = 0
    cut0 = 0
    # first, construct composite dataframe
    df2 = p1[["id"]]
    df2["id2"] = p2["id"]
    df2["y"] = y
    df2["match"] = oof_preds.astype("float32")
    df2 = df2.merge(train[["id", "m_true"]], on="id", how="left")  # bring in m_true
    cut2 = 0.8  # hardcode for now
    for cut1 in [0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
        # select matching pairs
        cut = cut1
        if (cut1 == 0):
            # true match, for max cv assessment only -
            #  this is CV if lgb model predicts a perfect answer
            matches = df2[["id", "id2", "match"]].loc[df2["y"] == 1].reset_index(drop=True)
        else:
            matches = (
                df2[["id", "id2", "match"]].loc[df2["match"] > cut].reset_index(drop=True)
            )  # call it a match if p > cut

        # construct POI from pairs
        poi_di = {}  # maps each id to POI
        poi = 0
        id1, id2, preds = (
            matches["id"].to_numpy(),
            matches["id2"].to_numpy(),
            matches["match"].to_numpy(),
        )
        for i in range(matches.shape[0]):
            i1 = id1[i]
            i2 = id2[i]
            if i1 in poi_di:  # i1 is already in dict - assign i2 to the same poi
                poi_di[i2] = poi_di[i1]
            else:
                if i2 in poi_di:  # i2 is already in dict - assign i1 to the same poi
                    poi_di[i1] = poi_di[i2]
                else:  # new poi, assign it to both
                    poi_di[i1] = poi
                    poi_di[i2] = poi
                    poi += 1

        # check for split groups and merge them
        for k in range(20):
            j = 0
            for i in range(matches.shape[0]):
                i1 = id1[i]
                i2 = id2[i]
                pred = preds[i]
                if (
                    poi_di[i2] != poi_di[i1] and pred > cut2
                ):  # 2 different groups - put them all in lower one
                    m = min(poi_di[i2], poi_di[i1])
                    poi_di[i1] = m
                    poi_di[i2] = m
                    j += 1
            if j == 0:
                break

        # construct list of id/poi pairs
        m2 = pd.DataFrame(matches["id"].append(matches["id2"], ignore_index=True))
        m2 = m2.drop_duplicates()
        m2["poi"] = m2[0].map(poi_di)

        # predicted true groups
        m2 = m2.sort_values(by=["poi", 0]).reset_index(drop=True)
        ids, pois = m2[0].to_numpy(), m2["poi"].to_numpy()
        poi0 = pois[0]
        id0 = ids[0]
        di_poi = {}  # this maps POI to list of all ids that belong to it
        for i in range(1, m2.shape[0]):
            if pois[i] == poi0:  # this id belongs to the same POI as prev id - add it to the list
                id0 = (
                    str(id0) + " " + str(ids[i])
                )  # id0 is list of all ids that belong to current POI
            else:
                di_poi[poi0] = str(id0) + " "  # need to have trailing space in m_true
                poi0 = pois[i]
                id0 = ids[i]
        di_poi[poi0] = str(id0) + " "  # need to have trailing space in m_true
        m2["m2"] = m2["poi"].map(di_poi)  # this is the list of all matches
        m2.columns = ["id", "poi", "m2"]

        # bring predictions into final result
        p1_tr = train[["id", "m_true"]].merge(m2[["id", "m2"]], on="id", how="left")
        p1_tr["m2"] = p1_tr["m2"].fillna("missing")
        idx = p1_tr["m2"] == "missing"
        p1_tr["m2"].loc[idx] = (
            p1_tr["id"].astype("str").loc[idx] + " "
        )  # fill missing values with id - those correspond to 1 id per poi

        # compare to true groups
        ii = np.zeros(p1_tr.shape[0], dtype=np.int32)
        x1, x2 = p1_tr["m_true"].to_numpy(), p1_tr["m2"].to_numpy()
        for i in range(p1_tr.shape[0]):
            ii[i] = cci(x1[i], x2[i])
        p1_tr["intersection"] = ii
        p1_tr["len_pred"] = p1_tr["m2"].apply(lambda x: x.count(" "))
        p1_tr["len_true"] = p1_tr["m_true"].apply(lambda x: x.count(" "))
        p1_tr["union"] = p1_tr["len_true"] + p1_tr["len_pred"] - p1_tr["intersection"]
        cv = (p1_tr["intersection"] / p1_tr["union"]).mean()
        if cv > cv0 or cut0 == 0:  # always overwrite 0, that was only a max assessment
            cv0 = cv
            cut0 = cut1

        if (y == oof_preds).all():
            print("Highest reachable IoU :", np.round(cv, 4))
            return
        print(cut1, "CV***", np.round(cv, 4))

    print(
        "best cut is",
        cut0,
        "best CV is",
        np.round(cv0, 4),
        "*************************************************************************",
    )
    return cut0, cv0


def print_infos(p1, p2=None, N_TO_FIND=360000):
    p1 = p1.reset_index(drop=True)

    if p2 is not None:
        p2 = p2.reset_index(drop=True)
        p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(
            np.int8
        )
        p1.loc[p1["id"] == p2["id"]]["y"] = 0

        p2["y"] = p1["y"]

        ps = pd.concat([p1, p2], 0)
    else:
        ps = p1

    ps = ps[ps["y"] == 1].reset_index()

    clusts = (
        ps[["id", "point_of_interest"]]
        .groupby("point_of_interest")
        .agg(lambda x: list(set(list(x))))
        .reset_index()
    )
    found = clusts["id"].apply(lambda x: len(x)).sum()

    print(f"Number of candidates : {numerize(len(p1))}")
    print(f"Proportion of positive candidates: {p1.y.mean() * 100:.2f}%")
    print(f"Proportion of found matches: {found / N_TO_FIND * 100:.2f}%")


##########################
#          LCS           #
##########################


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
                matrix[aIndex, bIndex] = max(matrix[aIndex - 1, bIndex], matrix[aIndex, bIndex - 1])
    return matrix[stringA.shape[0], stringB.shape[0]]


def lcs(stringA, stringB):
    a = np.frombuffer(stringA.encode(), dtype=np.int8)
    b = np.frombuffer(stringB.encode(), dtype=np.int8)
    matrix = np.zeros([1 + a.shape[0], 1 + b.shape[0]], dtype=np.int8)
    return lcs_n(a, b, matrix)


# feature: count of distinct substrings of length >=X in both names
def cc_lcs(str1, str2, x):
    c = 0  # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
        if d.size < x:  # no more X+ substrings - exit
            return c
        c += 1
        # remove found substring
        if d.a > 0:
            str1 = str1[:d.a] + str1[d.a+d.size:]
        else:
            str1 = str1[d.size:]
        if d.b > 0:
            str2 = str2[:d.b] + str2[d.b+d.size:]
        else:
            str2 = str2[d.size:]


# feature: total length of distinct substrings of length >=X in both names
def ll_lcs(str1, str2, x):
    c = 0  # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
        if d.size < x:  # no more X+ substrings - exit
            return c
        c += d.size
        # remove found substring
        if d.a > 0:
            str1 = str1[:d.a] + str1[d.a+d.size:]
        else:
            str1 = str1[d.size:]
        if d.b > 0:
            str2 = str2[:d.b] + str2[d.b+d.size:]
        else:
            str2 = str2[d.size:]


##########################
#         TF-IDF         #
##########################


def top_n_idx_sparse(matrix, N_row_pick):
    """
    Renvoie les index des n plus grandes valeurs de chaque ligne d'une sparse matrix
    impose_valeur_differente : Impose (si possible) au moins une valeur non-maximale
    pour éviter d’ignorer un score maximal si trop d’élèments en ont un
    """

    top_n_idx = []
    i = 0
    # matrix.indptr = index du 1er élèment (non nul) de chaque ligne
    for gauche, droite in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(
            N_row_pick[i], droite - gauche
        )  # pour gérer les cas où n est plus grand que le nombre de valeurs non-nulles de la ligne
        index = matrix.indices[
            gauche + np.argpartition(matrix.data[gauche:droite], -n_row_pick)[-n_row_pick:]
        ]
        # Ajout des indexs trouvés
        top_n_idx.append(index[::-1])
        i += 1
    return top_n_idx


def vectorisation_similarite(corpus_A, thr=0.3):
    """Renvoie un dataframe avec les paires de libellés les plus similaires
    au sens TF-IDF et Jaro-Winkler, entre A et B."""

    # =================================
    # ETAPE 0 : Forçage en string pour éviter les erreurs, suppression des doublons
    # et suppression des espaces en préfixe/suffixe
    corpus_A = [str(x).strip().lower() for x in corpus_A]
    corpus_B = corpus_A.copy()

    # =================================
    # ÉTAPE 1 : Vectorisation du corpus
    vect = TfidfVectorizer()  # min_df=1, stop_words="english"
    tfidf_A = vect.fit_transform(
        corpus_A
    )  # Pas besoin de normaliser par la suite : le Vectorizer renvoie un tf-idf normalisé
    tfidf_B = vect.transform(corpus_B)  # Utilisation de la normalisation issue de A
    pairwise_similarity = tfidf_A * tfidf_B.T
    # Sparse matrice (les élèments nuls ne sont pas notés) de dimension égale
    # aux nombres de lignes dans les documents
    N, M = pairwise_similarity.shape  # taille de la matrice

    # =======================================================
    # ÉTAPE 2 : Calcul des indices des n plus grandes valeurs

    # Calcul des élèments non-nuls de pairwise_similarity
    Elt_non_nuls = np.split(
        pairwise_similarity.data[pairwise_similarity.indptr[0]: pairwise_similarity.indptr[-1]],
        pairwise_similarity.indptr[1:-1],
    )

    # Calcul du nb d'élèments à checker : tous les bons scores
    # OU les meilleurs scores AVEC au moins nb_best_score
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
        print("  ! La taille de la matrice est de {} MB.".format(taille_MB))

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


##########################
#        Haversine       #
##########################


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


##########################
#         Ratios         #
##########################


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
    for l in range(m, 0, -1):  # noqa
        if y[:l] in x or x[:l] in y:
            return l
    return 0


def pi1(x, y):  # pi=partial intersection: check if first N letters are in the other
    if len(x) < 4 or len(y) < 4:
        return 0
    if y[:4] in x or x[:4] in y:  # hardcode 4 here - for now
        return 1
    else:
        return 0


def pi2(x, y):  # is ending substring of x in y?
    m = min(len(x), len(y))
    for l in range(m, 0, -1):  # noqa
        if y[-l:] in x or x[-l:] in y:
            return l
    return 0


def Compute_Mdist_Mindex(matrix, nb_max_candidates=1000, thr_distance=None):
    nb_max_candidates = min(nb_max_candidates, len(matrix))
    latitudes = matrix[:, 0]  # * np.pi / 180
    longitudes = matrix[:, 1]  # * np.pi / 180
    Z = tuple(zip(latitudes, longitudes))  # latitude, longitude

    tree = spatial.KDTree(Z)
    M_dist, M_index = (tree.query(Z, min(nb_max_candidates, len(matrix))))

    if thr_distance is None:
        return M_dist, M_index

    # Threshold filter
    Nb_matches_potentiels = []
    for i in range(len(M_index)):
        n = len([d for d in M_dist[i] if d <= thr_distance])
        Nb_matches_potentiels.append(n)
    M_dist = [m[:Nb_matches_potentiels[i]] for i, m in enumerate(M_dist)]
    M_index = [m[:Nb_matches_potentiels[i]] for i, m in enumerate(M_index)]

    return M_dist, M_index


def create_address(address, city):
    if address == "nan" or len(address) <= 1:
        return "nan"
    elif city == "nan" or len(city) <= 1:
        return "nan"
    else:
        return address.lower().strip() + "-" + city.lower().strip()


def find_potential_matchs(row, work_names, work_phones, work_address):
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
