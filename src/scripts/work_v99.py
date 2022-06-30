import gc
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import warnings
import Levenshtein
import difflib
import sys
from numba import jit
from unidecode import unidecode
import re

warnings.simplefilter('ignore')
start_time = time.time()
random.seed(13)



# read data
train = pd.read_pickle('input/train.pkl')

train['id'] = train['id'].astype('category').cat.codes # turn id into ints to save spacetime - for train only!!!
print('finished reading data', int(time.time() - start_time), 'sec')
path = ''

# select 1 country only for testing - GB, 26K records
#train = train.loc[train['country']=='GB'].reset_index(drop=True)


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





#################################################################################
## TF-IDF function
#################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sys import getsizeof

def top_n_idx_sparse(matrix, N_row_pick):
    ''' 
    Renvoie les index des n plus grandes valeurs de chaque ligne d'une sparse matrix 
    impose_valeur_differente : Impose (si possible) au moins une valeur non-maximale pour éviter d’ignorer un score maximal si trop d’élèments en ont un
    '''
    
    top_n_idx = []
    i = 0
    # matrix.indptr = index du 1er élèment (non nul) de chaque ligne
    for gauche, droite in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(N_row_pick[i], droite-gauche) # pour gérer les cas où n est plus grand que le nombre de valeurs non-nulles de la ligne
        index = matrix.indices[gauche + np.argpartition(matrix.data[gauche:droite], -n_row_pick)[-n_row_pick:]]
        # Ajout des indexs trouvés
        top_n_idx.append(index[::-1])
        i += 1
    return top_n_idx

def vectorisation_similarite(corpus_A, thr=0.3):
    ''' Renvoie un dataframe avec les paires de libellés les plus similaires au sens TF-IDF et Jaro-Winkler, entre A et B. 
    '''
    
    # =================================
    # ETAPE 0 : Forçage en string pour éviter les erreurs, suppression des doublons et suppression des espaces en préfixe/suffixe
    corpus_A = [str(x).strip().lower() for x in corpus_A]
    corpus_B = corpus_A.copy()
    
    # =================================
    # ÉTAPE 1 : Vectorisation du corpus
    vect = TfidfVectorizer() # min_df=1, stop_words="english"                                                                                                                                                                                                  
    tfidf_A = vect.fit_transform(corpus_A) # Pas besoin de normaliser par la suite : le Vectorizer renvoie un tf-idf normalisé   
    tfidf_B = vect.transform(corpus_B) # Utilisation de la normalisation issue de A 
    pairwise_similarity = tfidf_A * tfidf_B.T # Sparse matrice (les élèments nuls ne sont pas notés) de dimension égale aux nombres de lignes dans les documents
    N, M = pairwise_similarity.shape # taille de la matrice 
 
    # =======================================================
    # ÉTAPE 2 : Calcul des indices des n plus grandes valeurs

    # Calcul des élèments non-nuls de pairwise_similarity
    Elt_non_nuls = np.split(pairwise_similarity.data[pairwise_similarity.indptr[0]:pairwise_similarity.indptr[-1]], pairwise_similarity.indptr[1:-1])
    
    # Calcul du nb d'élèments à checker : tous les bons scores OU les meilleurs scores AVEC au moins nb_best_score
    Nb_elt_max_par_ligne = [len(np.argwhere((liste >= thr) | (liste == np.amax(liste))).flatten().tolist()) if liste.size > 0 else 0 for liste in Elt_non_nuls]

    # Taille de la matrice (dense) créée
    taille_MB = round(getsizeof(Elt_non_nuls) / (1024*1024)) # Taille en MB de la matrixe todense()
    if taille_MB > 10 :
        print('  /!\ La taille de la matrice est de {} MB.'.format(taille_MB))

    # Calcul des indices argmax dans la csr_matrix pairwise_similarity
    Ind_n_max = top_n_idx_sparse(pairwise_similarity, Nb_elt_max_par_ligne) # Calcul des indices des grandes valeurs
    
    # Récuparéation des valeurs TF-IDF
    Valeurs = []
    for i, Liste_index in enumerate(Ind_n_max) :
        values = []
        for idx in Liste_index :
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
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#################################################################################
#################################################################################




# detect language based on character set
def isEnglish(s):
    ss = "ª°⭐•®’—–™&\xa0\xad\xe2\xf0" # special characters
    s = str(s).lower()
    for k in range(len(ss)):
        s = s.replace(ss[k], "")
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        # not english; check it still not english if western european characters are removed
        ss = "éáñóüäýöçãõúíàêôūâşè"
        for k in range(len(ss)):
            s = s.replace(ss[k], "")
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return 3 # really not english
        else:
            return 2 # spanish/french?
    else:
        return 1 # english

train['lang'] = train['name'].apply(isEnglish).astype('int8')



# fill-in missing categories, based on words in name
Key_words_for_cat = pd.read_pickle(path+'dict_for_missing_cat.pkl')

def process(cat, split=' '):
    cat = [x for x in str(cat).split(split) if cat != '' and len(x)>=2]
    # Keep only letters
    cat = [re.sub(r'[^a-zA-Z]', ' ', x) for x in cat]
    # Delete multi space
    cat = [re.sub('\\s+', ' ', x).strip() for x in cat]
    return cat

# Function to fill missing categories
def find_cat(name):
    global Key_words_for_cat
    name_list = process(unidecode(str(name).lower()))
    for cat, wordlist in Key_words_for_cat.items() :
        if any(name_word in name_list for name_word in wordlist) :
            return cat
    return ''

train['categories'] = train['categories'].fillna('')
idx_missing_cat = train[train['categories'] == ''].index
train.loc[idx_missing_cat, 'categories'] = train.loc[idx_missing_cat, 'name'].fillna('').apply(find_cat)
print('finished filling-in missing categories', int(time.time() - start_time), 'sec') # NA cats drop by around 25%
del Key_words_for_cat, idx_missing_cat
gc.collect()





# pre-format data
train['point_of_interest'] = train['point_of_interest'].astype('category').cat.codes # turn POI into ints to save spacetime
train['latitude']  = train['latitude'].astype('float32')
train['longitude'] = train['longitude'].astype('float32')
# sorted by count in candidate training data
countries = ['US','ID','TR','JP','TH','RU','MY','BR','SG','PH','BE','KR','GB','MX','DE','FR','ES','CL','UA','IT','CA','AU','SA','CN','HK','FI','NL',
'TW','AR','GR','CZ','KW','AE','IN','CO','RO','VN','IR','HU','SE','PE','PL','LV','PT','EG','AT','ZA','CH','BY','PY','RS','CR','DK','BG','IE','VE',
'DO','MV','CY','MK','EE','NZ','PR','BN','HR','NO','SK','IL','EC','MD','PA','LT','GT','KH','QA','BH','AZ','GE','SV','TN','LK','JO','UY','KE','KZ',
'MQ','LB','MA','IS','HN','SI','MT','GU','ME','OM','BO','TT','LU','JM','PK','BD','XX','MN','AM','TM','MO','LA','NI','BA','KG','NG','BB','UZ','NP',
'BS','GH','AW','AL','TZ','IQ','IM','UG','MU','VI','MC','GP','SD','XK','KY','ZM','MP','MZ','AX','CU','BM','SY','ET','JE','BZ','SM','LC','GG','VA',
'BL','TC','CW','AD','RE','PF','SR','GI','BW','AO','HT','FJ','WS','GD','GF','KP','VC','RW','SC','MG','DZ','VG','SX','PS','AF','MF','AG','CM','CI',
'DM','CD','SN','LI','AQ','MW','TL','BT','CV','KN','BF','AI','SZ','ZW','AN','LY','NC','YE','SO','GA','EU','PM','BI','GL','GM','BV','NE','GW','TJ',
'BQ','GQ','BJ','TG','ST','VU','PG','PW','TO','SH','GY','SL','YT','FO','DJ','EH','SJ','LR','SS','ZZ']
c_di = {}
for i,c in enumerate(countries):# map train/test countries the same way
    c_di[c] = min(50, i + 1) # cap country at 50 - after that there are too few cases per country to split them
train['country']   = train['country'].fillna('ZZ').map(c_di).fillna(50).astype('int16') # new country maps to missing (ZZ)



# true groups - id's of the same POI; this is the answer we are trying to predict; use it for CV
train   = train.reset_index()
train   = train.sort_values(by=['point_of_interest', 'id']).reset_index(drop=True)
id_all  = np.array(train['id'])
poi_all = np.array(train['point_of_interest'])
poi0    = poi_all[0]
id0     = id_all[0]
di_poi  = {}
for i in range(1, train.shape[0]):
    if poi_all[i] == poi0:
        id0 = str(id0) + ' ' + str(id_all[i])
    else:
        di_poi[poi0]    = str(id0) + ' ' # need to have trailing space in m_true
        poi0            = poi_all[i]
        id0             = id_all[i]
di_poi[poi0] = str(id0) + ' ' # need to have trailing space in m_true
train['m_true'] = train['point_of_interest'].map(di_poi)
train = train.sort_values(by='index').reset_index(drop=True) # sort back to original order
train.drop('index', axis=1, inplace=True)
print('finished true groups', int(time.time() - start_time), 'sec')
a = train.groupby('point_of_interest').size().reset_index()
print('count of all matching pairs is:', (a[0]-1).sum()) # match:398,786 - minimum # of pairs to get correct result
print('count2 of all matching pairs is:', (a[0]*(a[0]-1)//2).sum(), int(time.time() - start_time), 'sec') # max # of pairs, including overlaps
del a, id_all, poi_all, di_poi
gc.collect()



# create grouped category - column 'category_simpl' ****************************************

# Save copy
train['name_svg'] = train['name'].copy()
train['categories_svg'] = train['categories'].copy()

# Clean name
train['name'] = train['name'].apply(lambda x : unidecode(str(x).lower()))

def replace_seven_eleven(text) :
    new = 'seven eleven'
    for sub in ['7/11', '7-11', '7-eleven'] :
        text = text.replace(sub+'#', new+' ')
        text = text.replace(sub+' ', new+' ')
        text = text.replace(sub, new)
    return text
train['name'] = train['name'].apply(lambda text : replace_seven_eleven(text))

def replace_seaworld(text) :
    new = 'seaworld'
    for sub in ['sea world'] :
        text = text.replace(sub, new)
    return text
train['name'] = train['name'].apply(lambda text : replace_seaworld(text))

def replace_mcdonald(text) :
    new = 'mac donald'
    for sub in ['mc donald', 'mcd ', 'macd ', 'mcd', 'mcdonald', 'macdonald', 'mc donalds', 'mac donalds'] :
        text = text.replace(sub, new)
    return text
train['name'] = train['name'].apply(lambda text : replace_mcdonald(text))

# Grouped categories
Cat_regroup = [['airport terminals',
  'airports',
  'airport services',
  'airport lounges',
  'airport food courts',
  'airport ticket counter',
  'airport trams',
  'airfields'],
 ['bus stations', 'bus stops'],
 ['opera houses', 'concert halls'],
['metro stations', 'tram stations', 'light rail stations', 'train stations'],
 ['auto garages',
  'auto workshops',
  'automotive shops',
  'auto dealerships',
  'motorcycle shops',
  'new auto dealerships'],
 ['hotels',
  'casinos',
  'hotel bars',
  'motels',
  'resorts',
  'residences',
  'inns',
  'hostels',
  'bed breakfasts'],
 ['bakeries',
  'borek places',
  'cupcake shops',
  'bagel shops',
  'breakfast spots',
  'gozleme places'],
 ['college classrooms',
  'college labs',
  'college science buildings',
  'college arts buildings',
  'college history buildings',
  'college cricket pitches',
  'college communications buildings',
  'college academic buildings',
  'college quads',
  'college auditoriums',
  'college engineering buildings',
  'college math buildings',
  'college bookstores',
  'college technology buildings',
  'college libraries',
  'libraries',
  'college football fields',
  'college administrative buildings',
  'general colleges universities',
  'universities',
  'community colleges',
  'high schools',
  'student centers',
  'college residence halls',
  'schools',
  'private schools'],
 ['movie theaters', 'film studios', 'indie movie theaters', 'multiplexes'],
 ['emergency rooms',
  'hospitals',
  'medical centers',
  'hospital wards',
  'medical supply stores',
  'physical therapists',
  'maternity clinics',
  'medical labs',
  'doctor s offices'],
 ['baggage claims', 'general travel', 'toll plazas'],
 ['cafes',
  'dessert shops',
  'donut shops',
  'coffee shops',
  'ice cream shops',
  'corporate coffee shops',
  'coffee roasters'],
 ['hockey arenas', 'basketball stadiums', 'hockey fields', 'hockey rinks'],
 ['buildings',
  'offices',
  'coworking spaces',
  'insurance offices',
  'banks',
  'campaign offices',
  'trailer parks',
  'atms'],
 ['capitol buildings', 'government buildings', 'police stations'],
 ['beach bars', 'beaches', 'surf spots', 'nudist beaches'],
 ['asian restaurants',
  'shabu shabu restaurants',
  'noodle houses',
  'chinese restaurants',
  'malay restaurants',
  'sundanese restaurants',
  'cantonese restaurants',
  'chinese breakfast places',
  'ramen restaurants',
  'indonesian restaurants',
  'satay restaurants',
  'javanese restaurants',
  'padangnese restaurants',
  'indonesian meatball places'],
 ['historic sites',
  'temples',
  'mosques',
  'spiritual centers',
  'monasteries',
  'churches',
  'history museums',
  'buddhist temples',
  'mountains'],
 ['bars', 'pubs'],
 ['gyms',
  'gyms or fitness centers',
  'gymnastics gyms',
  'gym pools',
  'yoga studios',
  'badminton courts',
  'courthouses'],
 ['fast food restaurants', 'burger joints', 'fried chicken joints'],
 ['nail salons',
  'salons barbershops',
  'perfume shops',
  'department stores',
  'cosmetics shops'],
 ['alternative healers', 'health beauty services', 'chiropractors', 'acupuncturists'],
 ['grocery stores', 'health food stores', 'supermarkets'],
 ['boutiques', 'clothing stores'],
 ['elementary schools', 'middle schools'],
 ['electronics stores', 'mobile phone shops'],
 ['convenience stores', 'truck stops', 'gas stations'],
 ['theme park rides attractions', 'theme parks'],
 ['outlet malls', 'shopping malls', 'adult boutiques', 'shopping plazas'],
 ['farmers markets', 'markets'],
 ['general entertainment', 'paintball fields'],
 ['som tum restaurants', 'thai restaurants'],
 ['piers', 'ports'],
 ['rugby stadiums', 'soccer stadiums', 'stadiums', 'soccer fields'],
 ['lounges', 'vape stores'],
 ['massage studios', 'spas'],
 ['racecourses', 'racetracks'],
 ['men s stores', 'women s stores'],
 ['american restaurants', 'tex mex restaurants'],
 ['japanese restaurants', 'sushi restaurants'],
 ['indian restaurants', 'mamak restaurants'],
 ['baseball fields', 'baseball stadiums'],
 ['tennis courts', 'tennis stadiums'],
 ['drugstores', 'pharmacies'],
 ['city halls', 'town halls'],
 ['ski areas', 'ski chalets', 'ski lodges'],
 ['lakes', 'reservoirs'],
 ['greek restaurants', 'tavernas'],
 ['hills', 'scenic lookouts'],
 ['college soccer fields',
  'college stadiums',
  'college hockey rinks',
  'college tracks',
  'college basketball courts'],
 ['furniture home stores', 'mattress stores', 'lighting stores'],
 ['recruiting agencies', 'rehab centers'],
 ['art museums', 'art studios', 'art galleries', 'museums', 'history museums'],
 ['outdoor supply stores', 'sporting goods shops'],
 ['czech restaurants', 'restaurants'],
 ['street fairs', 'street food gatherings'],
 ['canal locks', 'canals'],
 ['sake bars', 'soba restaurants'],
 ['bookstores', 'newsagents', 'newsstands', 'stationery stores'],
 ['other great outdoors', 'rafting spots'],
 ['manti places', 'turkish restaurants'],
 ['shoe repair shops', 'shoe stores'],
 ['photography labs', 'photography studios'],
 ['bowling alleys', 'bowling greens'],
 ['dry cleaners', 'laundry services'],
 ['cigkofte places', 'kofte places'],
 ['strip clubs', 'other nightlife', 'gay bars', 'nightclubs', 'rock clubs'],
 ['dog runs', 'parks', 'forests', 'rv parks', 'playgrounds'],
 ['convention centers', 'event spaces', 'conventions'],
 ['cruise ships', 'harbors marinas', 'piers', 'boats or ferries'],
 ['italian restaurants', 'pizza places'],
 ['law schools', 'lawyers'],
 ['bubble tea shops', 'tea rooms'],
 ['monuments landmarks', 'outdoor sculptures'],
 ['beer bars', 'beer stores', 'beer gardens', 'breweries', 'brasseries'],
 ['kebab restaurants', 'steakhouses'],
 ['real estate offices',
  'rental services',
  'rental car locations',
  'residential buildings apartments condos'],
 ['golf courses', 'mini golf courses'],
 ['food drink shops', 'food services', 'food stands', 'food trucks'],
 ['salad places', 'sandwich places', 'shawarma places'],
 ['ski chairlifts', 'ski trails', 'apres ski bars', 'skate parks'],
 ['wine shops', 'wineries'],
 ['flea markets', 'floating markets'],
 ['burrito places', 'taco places'],
 ['pet services', 'pet stores', 'veterinarians'],
 ['music festivals', 'music venues', 'music stores', 'music schools'],
 ['irish pubs', 'pie shops'],
 ['zoo exhibits', 'exhibits', 'zoos'],
 ['general travel', 'bridges'],
 ['sporting goods shops', 'athletics & sports', 'hunting supplies'],
 ['optical shops', 'eye doctors'],
 ['home services & repairs', 'other repair shops'],
]

import re
def process_text(text):
    text = unidecode(text.lower())
    res = ' '.join([re.sub(r'[^a-zA-Z]', ' ', x).strip() for x in text.split()])
    return re.sub('\\s+', ' ', res).strip()

def simplify_cat(categories) :
    global Cat_regroup
    categories = str(categories).lower()
    if categories in ('', 'nan') :
        return -1
    for cat in categories.split(',') :
        cat = process_text(cat)
        for i, Liste in enumerate(Cat_regroup) :
            if any(cat==x for x in Liste) :
                return i+1
    else : 
        return 0

train['category_simpl'] = train['categories'].astype(str).apply(lambda text : simplify_cat(text)).astype('int16')

print("Simpl categories found :", len(train[train['category_simpl'] > 0]), "/", len(train))

# Go back to initial columns
train['name'] = train['name_svg'].copy()
train['categories'] = train['categories_svg'].copy()
train.drop(['name_svg','categories_svg'], axis=1, inplace=True)




# remove all spaces, symbols, lower case
def st(x, remove_space=False):
    # turn to latin alphabet
    x = unidecode(str(x))
    # lower case
    x = x.lower()
    # remove symbols
    x = x.replace('"', "")
    ss = ",:;'/-+&()!#$%*.|\@`~^<>?[]{}_=\n"
    if remove_space :
        ss = " " + ss 
    for i in range(len(ss)):
        x = x.replace(ss[i], "")
    return x

def st2(x): # remove numbers - applies to cities only
    ss = " 0123456789"
    for i in range(len(ss)):
        x = x.replace(ss[i], "")
    return x

# Save names separated by spaces for tf-idf
train['categories_split'] = train['categories'].astype(str).apply(lambda x : [st(cat, remove_space=True) for cat in x.split(',')]).copy() # Create a new columns to split the categories
train['name_initial'] = train['name'].astype(str).apply(lambda x : x.lower()).copy()
train['name_initial_decode'] = train['name'].astype(str).apply(lambda x : st(x, remove_space=False)).copy()

solo_cat_scores = pd.read_pickle(path+'howmanytimes_groupedcat_are_paired_with_other_groupedcat.pkl') # link-between-grouped-cats
        
# Find the score of the categories
train['freq_pairing_with_other_groupedcat'] = train['category_simpl'].apply(lambda cat : solo_cat_scores[cat]).fillna(0)

solo_cat_scores = pd.read_pickle(path+'solo_cat_score.pkl') # link-between-categories - 1858 values

def apply_solo_cat_score(List_cat) :
    #global solo_cat_scores
    return max([solo_cat_scores[cat] for cat in List_cat])

# Find the score of the categories
train['cat_solo_score'] = train['categories_split'].apply(lambda List_cat : apply_solo_cat_score(List_cat)).fillna(0)

Dist_quantiles = pd.read_pickle(path+'Dist_quantiles_per_cat.pkl') # dist-quantiles-per-cat - 869 values

def apply_cat_distscore(List_cat) :
    #global Dist_quantiles
    q = np.array([Dist_quantiles[cat] for cat in List_cat if cat in Dist_quantiles])
    if len(q) == 0 :
        return Dist_quantiles['']
    return np.max(q, axis=0)

# Find the scores
col_cat_distscores = ["Nb_multiPoi", "mean", "q25", "q50", "q75", "q90", "q99"]
train.loc[:, col_cat_distscores] = train['categories_split'].apply(apply_cat_distscore).to_list() # 'Nb_multiPoi', 'mean', 'q25', 'q50', 'q75', 'q90','q99'
for col in ['cat_solo_score', 'freq_pairing_with_other_groupedcat', 'Nb_multiPoi', 'mean', 'q25', 'q50', 'q75', 'q90', 'q99']:
    train[col] = train[col].astype('float32')


# remove some expressions from name*********************************************
def rem_expr(x):
    x = str(x)
    x = x.replace("™", "") # tm
    x = x.replace("®", "")  # r
    x = x.replace("ⓘ", "") # i
    x = x.replace("©", "")  # c
    return x
train['name'] = train['name'].apply(rem_expr)

# drop abbreviations all caps in brakets for long enough names*******************
def rem_abr(x):
    x = str(x)
    if '(' in x and ')' in x:# there are brakets
        i = x.find('(')
        j = x.find(')')
        if j > i+1 and j-i < 10 and len(x) -(j-i) > 9: # remainder is long enough
            s = x[i+1:j]
            # clean it
            ss = " ,:;'/-+&()!#$%*.|`~^<>?[]{}_=\n"
            for k in range(len(ss)):
                s = s.replace(ss[k], "")
            if s == s.upper(): # all caps (and/or numbers)
                x = x[:i] + x[j+1:]
    return x
train['name'] = train['name'].apply(rem_abr)


def clean_nums(x):# remove st/nd/th number extensions
    words = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '0th', '1th', '2th', '3th', '4 th', '5 th', '6 th'
             , '7 th', '8 th', '9 th', '0 th', '1 th', '2 th', '3 th', '1 st', '2 nd', '3 nd']
    for word in words:
        x = x.replace(word, word[0])
    return x

def rem_words(x):# remove common words without much meaning
    words = ['the', 'de', 'of', 'da', 'la', 'a', 'an', 'and', 'at', 'b', 'el', 'las', 'los', 'no', 'di', 'by', 'le', 'del', 'in'
             , 'co', 'inc', 'llc', 'llp', 'ltd', 'on', 'der',' das', 'die']
    for word in words:
        x = x.replace(' ' + word + ' ', ' ') # middle
        if x[:len(word)+1] == word+' ': # start
            x = x[len(word)+1:]
        if x[-len(word)-1:] == ' '+word: # end
            x = x[:-len(word)-1]
    return x

# select capitals only, or first letter of each word (which could have been capital)
def get_caps_leading(name):
    name = unidecode(name)
    if name[:3].lower() == 'the': # drop leading 'the' - do not include it in nameC
        name = name[3:]
    name = rem_words(name) # remove common words without much meaning; assume they are always lowercase
    name = clean_nums(name) # remove st/nd/th number extensions
    name = [x for x in str(name).split(' ') if name != '' and len(x)>=2]
    # keep only capitals or first letters
    name = [re.findall(r'^[a-z]|[A-Z]', x) for x in name]
    # merge
    name = [''.join(x) for x in name]
    name = ''.join(name)
    return name.lower()
train['nameC'] = train['name'].fillna('').apply(get_caps_leading)

def clean_address(x):
    wwords = [['str', 'jalan', 'jl', 'st', 'street', 'ul', 'ulitsa', 'rue', 'rua', 'via'],
              ['rd', 'road'],
              ['ave', 'av', 'avenue', 'avenida'],
              ['hwy', 'highway'],
              ['fl', 'floor', 'flr'],
              ['blvd', 'boulevard', 'blv'],
              ['center', 'centre'],
              ['dr', 'drive'],
              ['mah', 'mahallesi'],
              ['ste', 'suite'],
              ['prosp', 'prospekt']]
    for words in wwords:
        for word in words[1:]:
            x = x.replace(' ' + word + ' ', ' ' + words[0] + ' ') # middle
            if x[:len(word)+1] == word+' ': # start
                x = x.replace(word + ' ', words[0] + ' ')
            if x[-len(word)-1:] == ' '+word: # end
                x = x.replace(' ' + word, ' ' + words[0])
    return x

for col in ['name', 'address', 'city', 'state', 'zip', 'url', 'categories']:
    train[col] = train[col].astype('str').apply(st) # keep spaces
    if col in ['name', 'address']:
        train[col] = train[col].apply(rem_words)
        train[col] = train[col].apply(clean_nums)
        if col == 'address':
            train['address'] = train['address'].apply(clean_address)
    train[col] = train[col].apply(lambda x: x.replace(' ', '')) # remove spaces

train['city'] = train['city'].apply(st2) # remove digits from cities
train['latitude']  = np.round(train['latitude'], 5).astype('float32')
train['longitude'] = np.round(train['longitude'], 5).astype('float32')
# for sorting - rounded coordinates
train['lat2'] = np.round(train['latitude'], 0).astype('float32')
train['lon2'] = np.round(train['longitude'], 0).astype('float32')
# for sorting - short name
train['name2'] = train['name'].str[:7]
print('finished pre-processing', int(time.time() - start_time), 'sec')


# support functions**************************************************************************************
# distance, in meters
def distance(lat1, lon1, lat2, lon2):
    lat1 = lat1 * 3.14 / 180.
    lon1 = lon1 * 3.14 / 180.
    lat2 = lat2 * 3.14 / 180.
    lon2 = lon2 * 3.14 / 180.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distance = 6373000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return distance

# partial intersection.
def pi(x, y):# is starting substring of x in y?
    m = min(len(x), len(y))
    for l in range(m,0,-1):
        if y[:l] in x or x[:l] in y:
            return l
    return 0

def pi1(x, y):# pi=partial intersection: check if first N letters are in the other
    if y[:4] in x or x[:4] in y: # hardcode 4 here - for now
        return 1
    else:
        return 0

def pi2(x, y):# is ending substring of x in y?
    m = min(len(x), len(y))
    for l in range(m,0,-1):
        if y[-l:] in x or x[-l:] in y:
            return l
    return 0





# clean name
name_di = {'uluslararası':'international', 'havaalani':'airport', 'havalimani':'airport', 'stantsiiametro':'metro', 'aeropuerto':'airport','seveneleven':'7eleven','kfckfc':'kfc','carefour':'carrefour','makdonalds':'mcdonalds','xingbake':'starbucks','mcdonaldss':'mcdonalds', 'kentuckyfriedchicken':'kfc', 'restoran':'restaurant', 'aiport':'airport', 'terminal':'airport', 'starbuckscoffee':'starbucks', '7elevenechewniielfewn':'7eleven', 'adtsecurityservices':'adt', 'ambarrukmoplaza':'ambarukmoplaza', 'attauthorizedretailer':'att','bandarjakarta':'bandardjakarta', 'dairyqueenaedriikhwiin':'dairyqueen', 'dunkindonut':'dunkin', 'dunkindonuts':'dunkin', 'dunkindonutsdnkndwnts':'dunkin','ionmoruguichuanzhuchechang':'ionmoruguichuan',
'tebingkaraton':'tebingkeraton', 'tebingkraton':'tebingkeraton', '711':'7eleven', 'albertsonspharmacy':'albertsons', 'applebeesgrillbar':'applebees', 'attstore':'att', 'autozoneautoparts':'autozone','awrestaurant':'aw', 'chilisgrillbar':'chilis', 'creditrepairservices':'creditrepair', 'dominospizza':'dominos', 'firestonecompleteautocare':'firestone', 'flyingjtravelcenter':'flyingj',
'libertytaxservice':'libertytax', 'mcdonald':'mcdonalds', 'papajohnspizza':'papajohns', 'pepboysautopartsservice':'pepboys', 'piatiorochka':'piaterochka', 'pilottravelcenters':'pilottravelcenter',
'sainsburyslocal':'sainsburys', 'sberbankrossii':'sberbank', 'shellgasstation':'shell', 'sprintstore':'sprint', 'strbks':'starbucks', 'starbucksreserve':'starbucks', 'usbankbranch':'usbank','verizonauthorizedretailercellularsales':'verizon', 'verizonwireless':'verizon', 'vodafonecepmerkezi':'vodafone', 'vodafoneshop':'vodafone', 'walmartneighborhoodmarket':'walmart', 'walmartsupercenter':'walmart',
'wellsfargobank':'wellsfargo', 'zaxbyschickenfingersbuffalowings':'zaxbys', 'ashleyfurniturehomestore':'ashleyhomestore','ashleyfurniture':'ashleyhomestore'}
# fix some misspellings
for w in name_di.keys():
    train['name'] = train['name'].apply(lambda x: x.replace(w, name_di[w]))

# new code from V *************************************************************
# Group names
name_groups = pd.read_pickle(path+'name_groups.pkl')    
# Translation
trans = {}
for best, group in name_groups.items():
    for n in group :
        trans[n] = best
train['name_grouped'] = train['name'].apply(lambda n : trans[n] if n in trans else n)
print(f"Grouped names : {len(train[train['name_grouped'] != train['name']])}/{len(train)}.")
train['name'] = train['name_grouped'].copy()
train = train.drop(columns = ['name_grouped'])
del name_groups, trans
gc.collect()

# cap length at 76
train['name'] = train['name'].str[:76]
# eliminate some common words that do not change meaning
for w in ['center']:
    train['name'] = train['name'].apply(lambda x: x.replace(w, ""))
train['name'].loc[train['name']=='nan'] = ''
#walmart
train['name'] = train['name'].apply(lambda x: 'walmart' if 'walmart' in x else x)
#carrefour
train['name'] = train['name'].apply(lambda x: 'carrefour' if 'carrefour' in x else x)
# drop leading 'the' from name
idx = train['name'].str[:3] == 'the' # happens 17,712 times = 1.5%
train['name'].loc[idx] = train['name'].loc[idx].str[3:]
print('finished cleaning names', int(time.time() - start_time), 'sec')

# clean city
city_di = {'alkhubar': 'alkhobar', 'khobar': 'alkhobar', 'muratpasa': 'antalya', 'antwerpen': 'antwerp', 'kuta': 'badung', 'bandungregency': 'bandung', 'bengaluru': 'bangalore', 'bkk': 'bangkok',
'krungethphmhaankhr': 'bangkok', 'pattaya': 'banglamung', 'sathon': 'bangrak', 'silom': 'bangrak', 'beijingshi': 'beijing', 'beograd': 'belgrade', 'ratchathewi': 'bangkok', 'brussels': 'brussel',
'bruxelles': 'brussel', 'bucuresti': 'bucharest', 'capitalfederal': 'buenosaires', 'busangwangyeogsi': 'busan', 'cagayandeorocity': 'cagayandeoro', 'cebucity': 'cebu', 'mueangchiangmai': 'chiangmai', 'qianxieshi': 'chiba', 'qiandaitianqu': 'chiyoda', 'zhongyangqu': 'chuo', 'sumedang': 'cikeruh', 'mexico': 'ciudaddemexico', 'mexicocity': 'ciudaddemexico', 'mexicodf': 'ciudaddemexico', 'koln': 'cologne', 'kobenhavn': 'copenhagen', 'osaka': 'dabanshibeiqu', 'jakarta': 'dkijakarta', 'dnipropetrovsk': 'dnepropetrovsk', 'frankfurtammain': 'frankfurt', 'fukuoka': 'fugangshi', 'minato': 'gangqu', 'moscow': 'gorodmoskva', 'moskva': 'gorodmoskva', 'sanktpeterburg': 'gorodsanktpeterburg', 'spb': 'gorodsanktpeterburg', 'hoankiem': 'hanoi', 'yokohama': 'hengbangshi', 'hochiminhcity': 'hochiminh', 'shouye_': 'home_', 'krungethph': 'huaikhwang', 'konak': 'izmir', 'kocaeli': 'izmit', 'jakartacapitalregion': 'dkijakarta', 'southjakarta': 'jakartaselatan', 'shanghai': 'jingan', 'shanghaishi': 'jingan', 'kyoto': 'jingdushi', 'melikgazi': 'kayseri', 'kharkov': 'kharkiv', 'kiyiv': 'kiev', 'kyiv': 'kiev', 'paradise': 'lasvegas', 'lisbon': 'lisboa', 'makaticity': 'makati', 'mandaluyongcity': 'mandaluyong', 'milano': 'milan', 'mingguwushizhongqu': 'mingguwushi', 'nagoyashi': 'mingguwushi', 'munich': 'munchen', 'muntinlupacity': 'muntinlupa', 'pasaycity': 'pasay', 'pasigcity': 'pasig', 'samsennai': 'phayathai', 'praha': 'prague', 'santiagodechile': 'santiago', 'zhahuangshi': 'sapporo', 'seoulteugbyeolsi': 'seoul', 'shenhushizhongyangqu': 'shenhushi', 'shibuya': 'shibuiguqu', 'xinsuqu': 'shinjuku', 'sofiia': 'sofia', 'surakarta': 'solo', 'suwonsi': 'suweonsi', 'taguigcity': 'taguig', 'taipei': 'taibeishi', 'watthana': 'vadhana', 'wien': 'vienna', 'warszawa': 'warsaw', 'washingtondc': 'washington',
'surgutkhantymansiiskiiavtonomnyiokrugiugraaorossiiskaiafederatsiia':'surgut', 'newyorkcity':'newyork', 'newyorknyus':'newyork', 'ny':'newyork', 'nyc':'newyork',
'londongreaterlondon':'london', 'greaterlondon':'london', 'losangelescaus':'losangeles', 'dabanshibeiqu':'dabanshi', 'seoulsi':'seoul', 'kuwaitcity':'kuwait', 'bangkoknoi':'bangkok'}
for key in city_di.keys():
    train['city'].loc[train['city'] == key] = city_di[key]
# second pass
city_di2 = {'jakartaselatan':'dkijakarta','jakartapusat':'dkijakarta','jakartabarat':'dkijakarta','jakartautara':'dkijakarta','jakartatimur':'dkijakarta','saintpetersburg':'sanktpeterburg'}
for key in city_di2.keys():
    train['city'].loc[train['city'] == key] = city_di2[key]

# cap length at 38
train['city'] = train['city'].str[:38]
# eliminate some common words that do not change meaning
for w in ['gorod']:
    train['city'] = train['city'].apply(lambda x: x.replace(w, ""))
train['city'].loc[train['city']=='nan'] = ''
print('finished cleaning cities', int(time.time() - start_time), 'sec')

# clean address
train['address'].loc[train['address']=='nan'] = ''
# cap length at 99
train['address'] = train['address'].str[:99]
train['address'] = train['address'].apply(lambda x: x.replace('street', 'str'))

# clean state
# cap length at 33
train['state'] = train['state'].str[:33]
state_di = {'calif':'ca', 'jakartacapitalregion':'jakarta', 'moscow':'moskva', 'seoulteugbyeolsi':'seoul'}
for key in state_di.keys():
    train['state'].loc[train['state'] == key] = state_di[key]
train['state'].loc[train['state']=='nan'] = ''

# clean url
# cap length at 129
train['url'] = train['url'].str[:129]
train['url'].loc[train['url']=='nan'] = ''
idx = train['url'].str[:8] == 'httpswww'
train['url'].loc[idx] = train['url'].str[8:].loc[idx]
idx = train['url'].str[:7] == 'httpwww'
train['url'].loc[idx] = train['url'].str[7:].loc[idx]
idx = train['url'].str[:5] == 'https'
train['url'].loc[idx] = train['url'].str[5:].loc[idx]
idx = train['url'].str[:4] == 'http'
train['url'].loc[idx] = train['url'].str[4:].loc[idx]
train['url'].loc[train['url']=='nan'] = ''

# clean phone
def process_phone(text) :
    text = str(text)
    if text == 'nan' : return ''
    L = []
    for char in text :
        if char.isdigit() :
            L.append(char)
    res = ''.join(L)[-10:].zfill(10)
    if len(res) > 0 : 
        return res
    else :
        return text
train['phone'] = train['phone'].apply(lambda text : process_phone(text))
# all matches of last 9 digits look legit - drop leading digit
train['phone'] = train['phone'].str[1:]
# set invalid numbers to empty
idx = (train['phone'] == '000000000') | (train['phone'] == '999999999')
train['phone'].loc[idx] = ''

# clean categories
# cap length at 68
train['categories'] = train['categories'].str[:68]
train['categories'].loc[train['categories']=='nan'] = ''
cat_di = {'aiport':'airport', 'terminal':'airport'}
for key in cat_di.keys():
    train['categories'] = train['categories'].apply(lambda x: x.replace(key, cat_di[key]))
print('finished cleaning categories', int(time.time() - start_time), 'sec')




# translate some common words*******************************************************************************************************
# indonesia
idx = train['country']==2 # ID
id_di = {'bandarudara':'airport', 'bandara':'airport', 'ruang':'room', 'smanegeri':'sma', 'sman':'sma', 'gedung':'building', 'danau':'lake', 'sumatera':'sumatra', 'utara':'north', 'barat':'west', 'jawa':'java', 'timur':'east', 'tengah':'central', 'selatan':'south', 'kepulauan':'island'}

def id_translate(x):# translate, and move some new words to the beginning
    global id_di
    for k in id_di.keys():
        if k in x:
            if id_di[k] in ['north', 'west', 'east', 'central', 'south']: # these go in the front
                x = id_di[k] + x.replace(k, '')
            elif id_di[k] in ['building', 'lake', 'island']: # these go in the back
                x =  x.replace(k, '') + id_di[k]
            else:
                x = x.replace(k, id_di[k])
    return x

for col in ['name', 'address', 'city', 'state']:
    train[col].loc[idx] = train[col].loc[idx].apply(id_translate)
print('finished translating ID', int(time.time() - start_time), 'sec')


# translate russian words
dict_ru_en = pd.read_pickle(path+'dict_translate_russian.pkl')

def process_text(x):
    return  re.findall('\w+', x.lower().strip())

def translate_russian_word_by_word(text) :
    global dict_ru_en
    text = process_text(text)
    text = [dict_ru_en[word] if word in dict_ru_en else word for word in text]
    return " ".join(text)

idx = train['country'] == 6 # RU
for k in ['city', 'state', 'address', 'name'] :
    train.loc[idx, k] = train.loc[idx, k].astype(str).apply(translate_russian_word_by_word)
    train.loc[idx, k] = train.loc[idx, k].apply(lambda x : '' if x=='nan' else x)
del dict_ru_en

# match some identical names - based on analysis of mismatched names for true pairs
# soekarno-hatta international airport - Jakarta, ID
l1 = ['soekarnohattainternationalairport', 'soekarnohatta', 'soekarnohattaairport', 'airportsoekarnohatta', 'airportinternasionalsoekarnohatta', 'soekarnohattainternationalairportjakarta'
      , 'airportsoekarnohattajakarta', 'airportinternationalsoekarnohatta', 'internationalairportsoekarnohatta', 'soekarnohattaintlairport', 'airportsoetta', 'soettainternationalairport'
      , 'soekarnohattainternasionalairport', 'soekarnohattaairport','soekarnohattaairport','soetta','airportsukarnohatta', 'soekarnohattaintairport','soekarnohattaairportinternational','soekarnohattaairportjakarta'
      , 'airportsoekarno','jakartainternationalairportsoekarnohatta','airportsoekarnohattaindonesia','airportsukarnohattainternational','soekarnohattainternationalairportckg']
idx = train['country']==2 # ID - this is where this location is
train['name'] .loc[idx]= train['name'].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
l1 = ['kualanamuinternationalairport','kualanamuairport','airportkualanamu','kualanamuinternationalairportmedan','airportinternasionalkualanamu','kualanamointernationalairport','airportinternationalkualanamumedan','airportkualanamuinternationalairport'
      ,'kualanamuairportinternasional','kualanamuinternasionalairportmedan','kualanamuinternationl','airportkualanamumedan','internationalairportkualanamu','kualanamuiternationalairportmedan','airportkualanamumedanindonesia','kualamanuinternationalairportmedan']
train['name'] .loc[idx]= train['name'].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
l1 = ['ngurahraiinternationalairport','ngurahraiairport','igustingurahraiinternationalairport','dpsngurahraiinternationalairport','airportngurahraidenpasarbali','ngurahraiinternationalairportairport','airportngurahraiinternationalairport','ngurahraiinternatioanlairportbali']
train['name'] .loc[idx]= train['name'].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
l1 = ['adisuciptointernationalairport','airportadisucipto','adisuciptointernasionalairport','airportadisuciptoyogyakarta','adisutjiptoairportjogjakarta','airportadisutjipto']
train['name'] .loc[idx]= train['name'].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
# many names in one nested list
ll1 = [['starbucks','starbucksstarbucks','staarbakhs','starbuck','starbucksstaarbakhs','sturbucks','starbuckscoffe','starbaks','starbacks','starbuks','starbucksdrivethru'],
['walmart','wallmart','wallyworld'],
['mcdonalds','mcdonaldss','macdonalds','makdonals','macdonals','mcdonals','macdonald','mccafe','mcdonaldssmccafe','mcd','mcdonaldssdrivethru','mcdrive','mcdonaldssmaidanglao','mcdonaldssmkdwnldz','mcdonaldssdrive','makdak'],
['paylessshoesource','payless','paylessshoes'],
['savealot','savelot'],
['subway','subwaysbwy','subwayrestaurants'],
['burgerking','burguerking'],
['pizzahut','pizzahutdelivery','pizzahutexpress'],
['dunkin','dunkindnkndwnts'],
['firestone','firestoneautocare'],
['dominos','dominopizza'],
['uspostoffice','postoffice','unitedstatespostalservice','usps','unitedstatespostoffice'],
['jcpenney','jcpenny','jcpopticalpopupshop','jcpenneyoptical','jcpenneysalon','jcpennysalon','jcpenneyportraitstudio'],
['enterpriserentacar','enterprise','enterprisecarvanhire'],
['littlecaesarspizza','littlecaesars','littleceasars'],
['nettomarkendiscount','nettofiliale','netto'],
['kroger','krogerfuel','krogerpharmacy','krogermarketplace'],
['tgifridays','fridays'],
['ngurahraiinternationalairport','airportngurahrai'],
['tangkubanperahu','tangkubanparahu'],
['adisuciptointernationalairport','adisuciptoairport','adisutjiptoairport','adisutjiptointernationalairport'],
['kualanamuinternationalairport','kualanamuinternasionalairport'],
['esenlerotogar','esenlerotogari'],
['hydepark','hydeparklondon','londonhydepark','hidepark','haydpark'],
['tesco','tescoexpress'],
['gatwickairport','londongatwickairport','gatwick','englandgatwickairport'],
['heathrowairport','londonheathrowairport','heathrowairportlondon','heathrow','londonheathrowinternationalairport','heatrowairport','londonheathrowairportairport','londonheathrowairportengland','heathrowunitedkingdom','heatrow','lhrairport'],
['metroadmiralteiskaia','admiralteiskaia','stantsiiametroadmiralteiskaia','admiralteiskaiametro','metroadmiralteiskaiametroadmiralteyskaya','admiralteiskaiastmetro'],
['kfc','kfccoffee','kfckfc','kfcdrivethru'],
['cvspharmacy','cvs'],
['chasebank','chase'],
['tgifridays','tgifriday','tgif'],
['costacoffee','costa'],
['jcpenney','jcpennys'],
['santanderbank','santander','bancosantander'],
['coffeebean','coffeebeantealeaf','coffebeantealeaf'],
['antalyaairport','antalyainternationalairport','antalyaairportairport','antalyaairportayt','antalyadishatlarairporti'],
['esenbogaairport','ankaraesenbogaairport','esenbogaairportairporti','esenbogaairportairport','esenbogaairportankara','esenbogainternationalairport','ankaraesenbogainternationalairport'],
['sabihagokcenairport','istanbulsabihagokcenairport','sabihagokcen','sabihagokceninternationalairport','sabihagokcenuluslararasiairport','istanbulsabihagokcenuluslararasiairport','sabihagokcenairportdishatlar','sabihagokcendishatlar','istanbulsabihagokcen','istanbulsabihagokceninternationalairport','sabihagokceninternetionalairport']]
for l1 in ll1:
    train['name'] = train['name'].apply(lambda x: x if x not in l1[1:] else l1[0])

def replace_common_words(s):
    for x in ['kebap', 'kebab', 'kepab', 'kepap']:
        s = s.replace(x, 'kebab')
    for x in ['aloonaloon']: # center place in indonesian villages
        s = s.replace(x, "alunalun")
    for x in ['restoram']:
        s = s.replace(x, "restaurant")
    s = s.replace('internationalairport', 'airport')
    return s

train['name'] = train['name'].apply(replace_common_words)





# define cat2 (clean category with low cardinality)
# base it on address, name and catogories - after those have been cleaned (then do not need to include misspellings)
train['cat2'] = '' # init (left: 129824*)
all_words = {# map all words in 'values' to 'keys'. Apply to address, name and categories
    'store':        ['dollartree', 'circlek', 'apteka', 'bricomarche', 'verizon', 'relay', 'firestone', 'alfamart', 'walgreens', 'carrefour', 'gamestop', 'radioshack', 'ikea', 'walmart', '7eleven', 'bodega', 'market', 'boutiqu', 'store', 'supermarket', 'shop', 'grocer', 'pharmac'],
    'restaurant':   ['warunkupnormal', 'mado', 'dominos', 'solaria', 'bistro', 'food', 'shawarma', 'tearoom', 'meatball', 'soup', 'breakfast', 'bbq'
                     , 'sushi', 'ramen', 'noodle', 'burger', 'sandwich', 'cafe', 'donut', 'restaurant', 'coffeeshop', 'buffet'
                     , 'pizzaplace', 'diner', 'steakhouse', 'kitchen', 'foodcourt', 'baker', 'starbucks', 'dunkin', 'tacoplac', 'snackplac'],
    'fastfood':     ['teremok', 'chickfil', 'arbys', 'popeyes', 'chilis', 'dairyqueen', 'tacobell', 'wendys', 'burgerking', 'fastfood', 'kfc', 'subway', 'pizzahut', 'mcdonalds', 'friedchicken'],
    'school':       ['sororityhous', 'fraternity', 'college', 'school', 'universit', 'classroom', 'student'],
    'housing':      ['home', 'housing', 'residential', 'building', 'apartment', 'condo'],
    'bank':         ['creditunion', 'bank', 'atm'],
    'airport':      ['baggageclaim', 'airport', 'terminal', 'airline', 'baggagereclaim', 'concourse'],
    'venue':        ['photographystudio', 'bowlingalle', 'cineplex', 'cinema', 'ballroom', 'stadium', 'meetingroom', 'conference', 'convention', 'entertainment', 'venue'
                     , 'auditorium', 'multiplex', 'eventspace', 'opera', 'concert', 'theater', 'megaplex'],
    'museum':       ['museum', 'galler'],
    'church':       ['sacred', 'shrine', 'spiritual', 'mosque', 'temple', 'cathedral', 'church', 'christ'],
    'park':         ['zoo', 'park'],
    'bar':          ['sportclips', 'speakeas', 'buffalowildwing', 'brewer', 'pub', 'bar', 'nightclub', 'nightlife', 'lounge'],
    'station':      ['flyingj', 'pilottravel', 'shell', 'bikerent', 'rail', 'station', 'train', 'metro', 'bus', 'stantsiia'],
    'medical':      ['poliklinika', 'diagnos', 'veterinarian', 'emergencyroom', 'hospital', 'medical', 'doctor', 'dentist'],
    'gym':          ['clinic', 'wellnes', 'sportsclub', 'gym', 'fitnes', 'athletic'],
    'outdoor':      ['farm', 'bagevi', 'bridges', 'surf', 'dogrun', 'sceniclookout', 'campground', 'golfcours', 'forest', 'river', 'outdoor', 'beach', 'field', 'plaza'
                     , 'lake', 'playground', 'mountain', 'pool', 'basketballcourt', 'garden'],
    'office':       ['techstartup', 'hrblock', 'work', 'creditrepair', 'librari','coworkingspaces', 'office', 'service', 'lawyer', 'courthous', 'cityhall', 'notarius'],
    'carrental':    ['rentalcar', 'hertz', 'rentacar', 'aviscarrent', 'dollarrent', 'zipcar', 'autodealership', 'carwashes'],
    'hotel':        ['boardinghous', 'hostel', 'hotel', 'motel'],
    }

for col in ['address', 'categories', 'name']:
    for word in all_words.keys():
        words = all_words[word]
        for w in words:
            train['cat2'].loc[train[col].str.contains(w, regex=False)] = word

cat2_di = {'':0, 'restaurant':1, 'bar':2, 'store':3, 'housing':4, 'office':5, 'outdoor':6, 'station':7, 'medical':8, 'venue':9,
'hotel':10, 'school':11, 'church':12, 'park':13,'bank':14, 'airport':15, 'gym':16, 'museum':17, 'carrental':18, 'fastfood':19}
train['cat2'] = train['cat2'].map(cat2_di).astype('int16')
print('finished cat2', int(time.time() - start_time), 'sec')


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
dist_by_cat2 = {0:176, 1:61, 2:67, 3:95, 4:97, 5:90, 6:270, 7:126, 8:111, 9:121, 10:75, 11:92, 12:122, 13:205, 14:78, 15:1112, 16:85, 17:98, 18:59, 19:133}
# median by category_simpl
dist_by_category_simpl = {0:135,1:1108,2:80,3:89,4:280,5:52,6:78,7:79,8:81,9:153,10:97,11:165,12:67,13:95,14:80,15:107,16:679,17:65,18:149,19:39,20:90,21:100,22:84,23:119,24:81,25:84,26:80,27:76,28:56,29:143,30:227,31:132,32:92,33:31,34:147,35:146,36:65,37:70,38:312,39:75,40:83,41:64,42:50,43:131,44:57,45:71,46:113,47:448,48:1409,49:30,50:273,51:127,52:145,53:97,54:68,55:83,56:37,57:136,58:5351,59:97,60:97,61:739,62:31,63:66,64:83,65:78,66:79,67:87,68:87,69:231,70:129,71:612,72:131,73:144,74:85,75:111,76:37,77:72,78:93,79:361,80:234,81:67,82:408,83:109,84:77,85:44,86:76,87:88,88:30,89:136,90:302,91:152,92:66,93:52}



# feature: count of distinct substrings of length >=X in both names
def cc_lcs(str1, str2, x):
    c = 0 # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
        if d.size < x: # no more X+ substrings - exit
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
    c = 0 # init counter
    for i in range(100):
        # find longest substring
        d = difflib.SequenceMatcher(None, str1, str2).find_longest_match(0, len(str1), 0, len(str2))
        if d.size < x: # no more X+ substrings - exit
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

            

# include all col - regardless of the shift********************************************
# phone
p3 = train[['country', 'id', 'point_of_interest', 'phone', 'lon2']].copy()
p3 = p3.loc[p3['phone'] != ''].sort_values(by=['country', 'phone', 'lon2', 'id']).reset_index(drop=True)
idx1 = []
idx2 = []
d = p3['phone'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 11): # 11 = add 10 sets
        if i + j < p3.shape[0] and lcs(d[i], d[i+j]) >= 7: # accept <=3 digits off
            idx1.append(i)
            idx2.append(i+j)
p1 = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2 = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('phone match: added', p1.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# lat/lon, rounded to 2 x 4 digits = 22* meters square; there should not be too many false positives this close to each other
# do this in 4 blocks, shifted by 1/2 size, to avoid cut-offs
for s1 in [0, 5e-5]:
    for s2 in [0, 5e-5]:
        p3 = train[['country', 'id', 'point_of_interest', 'latitude', 'longitude']].copy()
        p3['latitude'] = np.round(s1 + .5 * p3['latitude'], 4) # rounded to 4 digits
        p3['longitude'] = np.round(s2 + .5 * p3['longitude'] / np.cos(p3['latitude'] * 3.14 / 180.), 4) # rounded to 4 digits
        p3 = p3.sort_values(by=['country', 'latitude', 'longitude', 'id']).reset_index(drop=True)
        idx1 = [] 
        idx2 = []
        lat, lon = p3['latitude'].to_numpy(), p3['longitude'].to_numpy()
        for i in range(p3.shape[0]-1):
            for j in range(1, 5): # 5 = add 4 sets
                if i + j < p3.shape[0] and lat[i] == lat[i+j] and lon[i] == lon[i+j]:
                    idx1.append(i)
                    idx2.append(i+j)
        p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
        p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
        # append
        p1 = p1.append(p1a, ignore_index=True)
        p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('lat/lon match: added', p1.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# url
p3 = train[['country', 'id', 'point_of_interest', 'url', 'lon2', 'lat2']].copy()
p3 = p3.loc[p3['url'] != ''].sort_values(by=['country', 'url', 'lon2', 'lat2', 'id']).reset_index(drop=True)
idx1 = []
idx2 = []
d = p3['url'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 2): # 2 = add 1 set
        if i + j < p3.shape[0] and ll_lcs(d[i], d[i+j], 3) >= 7: # ll_lcs(3) >= 7
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('url match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# categories
p3 = train[['country', 'id', 'point_of_interest', 'categories', 'lon2', 'lat2']].copy()
p3 = p3.loc[p3['categories'] != ''].sort_values(by=['country', 'categories', 'lon2', 'lat2', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
d = p3['categories'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 2): # 2 = add 1 set
        if i + j < p3.shape[0] and d[i][:4] == d[i+j][:4]:  # match on first 4 leters
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('categories match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# address
p3 = train[['country', 'id', 'point_of_interest', 'address', 'lon2', 'lat2']].copy()
p3 = p3.loc[p3['address'] != ''].sort_values(by=['country', 'address', 'lon2', 'lat2', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
d = p3['address'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 7): # 7 = add 6 sets
        if i + j < p3.shape[0] and lcs2(d[i], d[i+j]) >= 6: # lcs2 >= 6
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('address match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# name
p3 = train[['country', 'id', 'point_of_interest', 'name', 'lon2', 'lat2']].copy()
p3 = p3.loc[p3['name'] != ''].sort_values(by=['country', 'name', 'lon2', 'lat2', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
d = p3['name'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 4): # 4 = add 3 sets
        if i + j < p3.shape[0] and lcs2(d[i], d[i+j]) >= 5: # lcs2 >= 5
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('name match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# latitude
p3 = train[['country', 'id', 'point_of_interest', 'name', 'latitude']].copy()
p3 = p3.sort_values(by=['country', 'latitude', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
d = p3['latitude'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 21): # 21 = add 20 sets
        if i + j < p3.shape[0] and d[i] == d[i+j]: # exact match
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('lat match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# longitude
p3 = train[['country', 'id', 'point_of_interest', 'name', 'longitude']].copy()
p3 = p3.sort_values(by=['country', 'longitude', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
d = p3['longitude'].to_numpy()
for i in range(p3.shape[0]-1):
    for j in range(1, 21): # 21 = add 20 sets
        if i + j < p3.shape[0] and d[i] == d[i+j]: # exact match
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('lon match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')

# name1
p3 = train[['country', 'id', 'point_of_interest', 'name', 'latitude', 'longitude', 'categories']].copy()
# rounded coordinates
p3['latitude'] = np.round(p3['latitude'], 1).astype('float32')      # rounding: 1=10Km, 2=1Km
p3['longitude'] = np.round(p3['longitude'], 1).astype('float32')
p3 = p3.sort_values(by=['country', 'latitude', 'longitude', 'categories', 'id']).reset_index(drop=True)
idx1 = [] 
idx2 = []
names = p3['name'].to_numpy()
lon2 = p3['longitude'].to_numpy()
for i in range(p3.shape[0]-1):
    if i%100000 == 0:
        print(i, int(time.time() - start_time), 'sec')
    li = lon2[i]
    for j in range(1, min(300, p3.shape[0]-1-i)): # put a limit here - look at no more than X items
        if li != lon2[i+j]: # if lon matches, lat and country also match - b/c of sorting order
            break
        if lcs2(names[i], names[i+j]) >= 5: # lcs2 >= 5
            idx1.append(i)
            idx2.append(i+j)
p1a = p3[['id', 'point_of_interest']].loc[idx1].reset_index(drop=True)
p2a = p3[['id', 'point_of_interest']].loc[idx2].reset_index(drop=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
a = p1.groupby('id')['y'].sum().reset_index()
print('name1 match: added', p1a.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), 'records', int(time.time() - start_time), 'sec')
del d, names, lon2, p3, idx1, idx2, p1a, p2a, lat, lon
gc.collect()



# remove duplicate pairs
p12 = pd.concat([p1['id'], p2['id']], axis=1)
p12.columns = ['id','id2']
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12['id'] > p12['id2']
p12['t'] = p12['id']
p12['id'].loc[idx] = p12['id2'].loc[idx]
p12['id2'].loc[idx] = p12['t'].loc[idx]

p12 = p12.sort_values(by=['id', 'id2']).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=['id', 'id2'])

# also drop id == id2 - it may happen
p12 = p12.loc[p12['id'] != p12['id2']]
p1 = p1.loc[p12['index']].reset_index(drop=True)
p2 = p2.loc[p12['index']].reset_index(drop=True)
del p12, idx
gc.collect()
y = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
print('removed duplicates', p1.shape[0], (p1['point_of_interest'] == p2['point_of_interest']).sum(), int(time.time() - start_time), 'sec')
# get stats
p1['y'] = y
a = p1.groupby('id')['y'].sum().reset_index()
print('count of all matching pairs is:', np.minimum(1,a['y']).sum())



def substring_ratio(name1, name2) :
    N = (len(name1) + len(name2)) / 2
    if N == 0 : return 0
    substr = lcs2(name1, name2)
    return substr / N

def subseq_ratio(name1, name2) :
    N = (len(name1) + len(name2)) / 2
    if N == 0 : return 0
    substr = lcs(name1, name2)
    return substr / N



# sort to put similar points next to each other - for constructing pairs
sort = ['lat2', 'lon2', 'name2', 'latitude', 'city', 'cat2', 'name', 'address', 'country', 'id']
train = train.sort_values(by=sort).reset_index(drop=True)
train.drop(['lat2', 'lon2', 'name2'], axis=1, inplace=True) # these are no longer needed
print('finished sorting', int(time.time() - start_time), 'sec')

# construct pairs***********************************************************************************************************
cols = ['id', 'latitude', 'longitude', 'point_of_interest', 'name', 'category_simpl']
colsa = ['id', 'point_of_interest']
p1a = train[colsa].copy()
p2a = train[colsa].iloc[1:,:].reset_index(drop=True).copy()
p2a = p2a.append(train[colsa].iloc[0], ignore_index=True)
# append
p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)
# add more shifts, only for short distances or for partial name matches
for i, s in enumerate(range(2, 121)): # 121
    if s == 15: # resort by closer location after 15 shifts
        train['lat2'] = np.round(train['latitude'], 2).astype('float32') # 2 = 1 Km
        train['lon2'] = np.round(train['longitude'], 2).astype('float32')
        train = train.sort_values(by=['country', 'lat2', 'lon2', 'categories', 'city', 'id']).reset_index(drop=True)
        train.drop(['lat2', 'lon2'], axis=1, inplace=True)
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
    s2 = s # shift
    if i >= 13: # resorted data
        s2 = i - 12
    p2a = train[cols].iloc[s2:,:]
    p2a = p2a.append(train[cols].iloc[:s2,:], ignore_index=True)

    # drop pairs with large distances
    dist = distance(np.array(train['latitude']), np.array(train['longitude']), np.array(p2a['latitude']), np.array(p2a['longitude']))
    same_cat_simpl = ((train['category_simpl']==p2a['category_simpl']) & (train['category_simpl']>0))
    
    ii = np.zeros(train.shape[0], dtype=np.int8)
    x1, x2 = train['name'].to_numpy(), p2a['name'].to_numpy()
    for j in range(train.shape[0]):
        if pi1(x1[j], x2[j]):
            ii[j] = 1
        elif substring_ratio(x1[j], x2[j]) >= 0.65 :
            ii[j] = 1
        elif subseq_ratio(x1[j], x2[j]) >= 0.75 :
            ii[j] = 1
        elif len(x1[j])>=7 and len(x2[j])>=7 and x1[j].endswith(x2[j][-7:]) :
            ii[j] = 1
    # keep if dist < maxdist, or names partially match
    #idx = (dist < maxdist) | (ii > 0)
    idx = (dist < maxdist) | (ii > 0) | np.logical_and(same_cat_simpl, dist<train['q90']*900)
    
    p1 = p1.append(train[colsa].loc[idx], ignore_index=True)
    p2 = p2.append(p2a[colsa].loc[idx], ignore_index=True)
    if s < 10 or s%10 == 0:
        # get stats; overstated b/c dups are not excluded yet
        p1['y'] = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
        a = p1.groupby('id')['y'].sum().reset_index()
        print(i, maxdist, s, s2, p1.shape[0], p1['y'].sum(), np.minimum(1,a['y']).sum(), int(time.time() - start_time), 'sec')
    gc.collect()
del p1a, p2a, dist, idx
gc.collect()





# remove duplicate pairs
p12 = pd.concat([p1['id'], p2['id']], axis=1)
p12.columns = ['id','id2']
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12['id'] > p12['id2']
p12['t'] = p12['id']
p12['id'].loc[idx] = p12['id2'].loc[idx]
p12['id2'].loc[idx] = p12['t'].loc[idx]

p12 = p12.sort_values(by=['id', 'id2']).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=['id', 'id2'])

# also drop id == id2 - it may happen
p12 = p12.loc[p12['id'] != p12['id2']]
p1 = p1.loc[p12['index']].reset_index(drop=True)
p2 = p2.loc[p12['index']].reset_index(drop=True)
del p12, idx
gc.collect()
y = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
print('removed duplicates', p1.shape[0], (p1['point_of_interest'] == p2['point_of_interest']).sum(), int(time.time() - start_time), 'sec')
# get stats
p1['y'] = y
a = p1.groupby('id')['y'].sum().reset_index()
print('count of all matching pairs is:', np.minimum(1,a['y']).sum())



# Add close candidates
from scipy import spatial
def Compute_Mdist_Mindex(matrix, nb_max_candidates=1000, thr_distance=None) :
    
    nb_max_candidates = min(nb_max_candidates, len(matrix))
    Z = tuple(zip(matrix[:, 0], matrix[:, 1])) # latitude, longitude

    tree = spatial.KDTree(Z)
    M_dist, M_index = (tree.query(Z, min(nb_max_candidates, len(matrix))))
    
    if thr_distance is None : 
        return M_dist, M_index

    # Threshold filter
    Nb_matches_potentiels = []
    for i in range(len(M_index)) :
        n = len([d for d in M_dist[i] if d <= thr_distance])
        Nb_matches_potentiels.append(n)
    M_dist  = [m[:Nb_matches_potentiels[i]] for i, m in enumerate(M_dist)]
    M_index = [m[:Nb_matches_potentiels[i]] for i, m in enumerate(M_index)]
    
    return M_dist, M_index

New_candidates = []

#for country_ in range(2, len(countries)+1):
for country_ in [2, 3]:
    
    nb_max_candidates = 400
    new_cand = set()

    # Create matrix
    matrix = train[train['country']==country_].copy()
    if len(matrix)<=1 : break
    Original_idx = {i:idx for i, idx in enumerate(matrix.index)}
    
    # Find closest neighbours
    M_dist, M_index = Compute_Mdist_Mindex(matrix[['latitude', 'longitude']].to_numpy(),
                                           nb_max_candidates=nb_max_candidates)

    # Select candidates
    new_true_match = 0
    infos = matrix[['id', 'name', 'point_of_interest']].to_numpy()
        
    for idx1, (Liste_idx, Liste_val) in enumerate(zip(M_index, M_dist)):
        for idx2, dist in zip(Liste_idx, Liste_val):
            if idx1==idx2 : continue
                
            # Too far candidates
            if dist > 0.12 :
                break

            id1, id2 = infos[idx1, 0], infos[idx2, 0]
            name1, name2 = infos[idx1, 1], infos[idx2, 1]

            if pi1(name1, name2)==1 or substring_ratio(name1, name2) >= 0.5 :
                key = tuple(sorted([id1, id2]))
         
    # Add new candidates        
    New_candidates += [list(x) for x in new_cand]
    print(f"Country {country_} ({countries[country_-1]}) : {new_true_match}/{len(new_cand)} new cand added.")

# Add matches
size1 = len(p1)
Added_p1, Added_p2 = [], []
for idx1, idx2 in New_candidates :
    id1, id2   = train['id'].iat[idx1], train['id'].iat[idx2]
    poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
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
p12 = pd.concat([p1['id'], p2['id']], axis=1)
p12.columns = ['id','id2']
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12['id'] > p12['id2']
p12['t'] = p12['id']
p12['id'].loc[idx] = p12['id2'].loc[idx]
p12['id2'].loc[idx] = p12['t'].loc[idx]

p12 = p12.sort_values(by=['id', 'id2']).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=['id', 'id2'])

# also drop id == id2 - it may happen
p12 = p12.loc[p12['id'] != p12['id2']]
p1 = p1.loc[p12['index']].reset_index(drop=True)
p2 = p2.loc[p12['index']].reset_index(drop=True)
del p12, idx, matrix
gc.collect()
y = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
print('removed duplicates', p1.shape[0], (p1['point_of_interest'] == p2['point_of_interest']).sum(), int(time.time() - start_time), 'sec')
# get stats
p1['y'] = y
a = p1.groupby('id')['y'].sum().reset_index()
print('count of all matching pairs is:', np.minimum(1,a['y']).sum())




# candidates in initial Youri's solution
ID_to_POI = dict(zip(train['id'], train['point_of_interest']))
nb_true_matchs_initial = 0
Cand = {}
for i, (id1, id2) in enumerate(zip(p1['id'], p2['id'])) :
    key = f"{min(id1, id2)}-{max(id1, id2)}"
    Cand[key] = p1['y'].iloc[i]
    nb_true_matchs_initial += int(ID_to_POI[id1] == ID_to_POI[id2])



#################################################################################
## TF-IDF n°1 : airports
#################################################################################
# Vincent

far_cat_simpl = [1, 2]
thr_tfidf = 0.45

for col_name in ['name_initial_decode']:

    Names = train[train['category_simpl'].isin(far_cat_simpl + [-1])][col_name].copy() # add unknown categories
    
    def process_terminal(text) :
        for i in range(0, 30) :
            text = text.replace(f"terminal {i}", "")
            text = text.replace(f"terminal{i}", "")
            text = text.replace(f"t{i}", "")
        return text
    Names = Names.apply(process_terminal)

    # Drop stop words
    Names = Names.apply(lambda x : x.replace('airpord', 'airport'))
    Names = Names.apply(lambda x : x.replace('internasional', 'international'))
    Names = Names.apply(lambda x : x.replace('internacional', 'international'))
    for stopword in ['terminal', 'airport', 'arrival', 'hall', 'departure', 'bus stop', 'airways', 'checkin'] :
        Names = Names.apply(lambda x : x.replace(stopword+'s', ''))
        Names = Names.apply(lambda x : x.replace(stopword, ''))
    Names = Names.apply(lambda x : x.strip())
    Names = Names[Names.str.len() >= 2]

    Names_numrow = {i:idx for i, idx in enumerate(Names.index)} # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=thr_tfidf)

        # no self-matchs and retrieve the initial row number
        Tfidf_no_selfmatch = [[Names_numrow[i], [Names_numrow[x] for x in L if x != i]] for i, L in enumerate(Tfidf_idx)]
        Tfidf_no_selfmatch = [x for x in Tfidf_no_selfmatch if len(x[-1])>0]
        print("Nb cand tf-idf :", sum([len(L) for idx, L in Tfidf_no_selfmatch]))

        # Add matches
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, Liste_idx in Tfidf_no_selfmatch :
            id1, name1, lat1, lon1 = train['id'].iat[idx1], train['name'].iat[idx1], train['latitude'].iat[idx1], train['longitude'].iat[idx1]
            cat1, country1, cat_simpl1 = train['categories'].iat[idx1], train['country'].iat[idx1], train['category_simpl'].iat[idx1]
            for idx2 in Liste_idx :
                #if len(Liste_idx)>30 : continue
                if idx1 < idx2 :
                    id2, lat2, lon2 = train['id'].iat[idx2], train['latitude'].iat[idx2], train['longitude'].iat[idx2]
                    cat2, country2, cat_simpl2 = train['categories'].iat[idx2], train['country'].iat[idx2], train['category_simpl'].iat[idx2]
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    #same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='')
                    if key not in Cand and (cat_simpl1==1 or cat_simpl2==1) and (haversine(lat1,lon1,lat2,lon2)<=100 or "kualalumpur" in name1) :
                        poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
                        Cand[key] = int(poi1==poi2)
                        Added_p1.append([id1, poi1, int(poi1==poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1==poi2)

        p1 = p1.append(pd.DataFrame(Added_p1, columns=p1.columns)).reset_index(drop=True).copy()
        p2 = p2.append(pd.DataFrame(Added_p2, columns=p2.columns)).reset_index(drop=True).copy()
        print(f"Candidates added for tfidf n°1 (airports) : {len(p1)-size1}/{len(p1)}.")


#################################################################################
## TF-IDF n°2 : metro stations
#################################################################################
# Vincent

far_cat_simpl = [4]
thr_tfidf = 0.45
thr_distance = 100

for col_name in ['name_initial', 'name_initial_decode']:

    Names = train[train['category_simpl'].isin(far_cat_simpl)][col_name].copy() # add unknown categories

    # Drop stop words
    for stopword in ['stasiun', 'station', 'metro', '北改札', 'bei gai zha', 'stasiun'] :
        Names = Names.apply(lambda x : x.replace(stopword+'s', ''))
        Names = Names.apply(lambda x : x.replace(stopword, ''))
    Names = Names.apply(lambda x : x.strip())
    Names = Names[Names.str.len() > 2]

    Names_numrow = {i:idx for i, idx in enumerate(Names.index)} # Keep initial row number
    Names = Names.to_list()

    print(f"Len names : {len(Names)}.")

    # Tf-idf
    if 1 < len(Names) < 400000:
        Tfidf_idx, Tfidf_val = vectorisation_similarite(Names, thr=thr_tfidf)

        # no self-matchs and retrieve the initial row number
        Tfidf_no_selfmatch = [[Names_numrow[i], [Names_numrow[x] for x in L if x != i]] for i, L in enumerate(Tfidf_idx)]
        Tfidf_no_selfmatch = [x for x in Tfidf_no_selfmatch if len(x[-1])>0]
        print("Nb cand tf-idf :", sum([len(L) for idx, L in Tfidf_no_selfmatch]))

        # Add matches
        size1 = len(p1)
        Added_p1, Added_p2 = [], []
        for idx1, Liste_idx in Tfidf_no_selfmatch :
            id1, lat1, lon1, cat1, cat_simpl1 = train['id'].iat[idx1], train['latitude'].iat[idx1], train['longitude'].iat[idx1], train['categories'].iat[idx1], train['category_simpl'].iat[idx1]
            for idx2 in Liste_idx :
                if idx1 < idx2 :
                    id2, lat2, lon2, cat2, cat_simpl2 = train['id'].iat[idx2], train['latitude'].iat[idx2], train['longitude'].iat[idx2], train['categories'].iat[idx2], train['category_simpl'].iat[idx2]
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    if key not in Cand and haversine(lat1,lon1,lat2,lon2)<=thr_distance and (cat_simpl1 in far_cat_simpl or cat_simpl2 in far_cat_simpl) :
                        poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
                        Cand[key] = int(poi1==poi2)
                        Added_p1.append([id1, poi1, int(poi1==poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1==poi2)

        p1 = p1.append(pd.DataFrame(Added_p1, columns=p1.columns)).reset_index(drop=True).copy()
        p2 = p2.append(pd.DataFrame(Added_p2, columns=p2.columns)).reset_index(drop=True).copy()
        print(f"Candidates added for tfidf n°2 (metro stations) : {len(p1)-size1}/{len(p1)}.")




#################################################################################
## TF-IDF n°3a : for each countries (with initial unprocessed name)
#################################################################################
# Vincent

thr_tfidf_ = 0.5
thr_distance_ = 20
thr_distance_or_same_cat_ = 2

size = len(p1)

for country in [2, 3, 32] : #range(1, 30)
    
    # Reset parameters
    thr_tfidf = thr_tfidf_
    thr_distance = thr_distance_
    thr_distance_or_same_cat = thr_distance_or_same_cat_
    
    # Tune parameters for each country
    if country == 2 :
        thr_tfidf    = 1.1 # will impose to have same category
        thr_distance = 20
        thr_distance_or_same_cat = -1  # will impose to have same category
    elif country == 3 :
        thr_tfidf    = 0.6
        thr_distance = 10
    elif country == 32 :
        thr_tfidf    = 0.4
        thr_distance = 100 # no limit
        thr_distance_or_same_cat = 100 # no limit

    # List of names
    Names = train[train['country']==country]['name_initial'].copy()
    if len(Names) == 0 : break
        
    print()
    print("#"*20)
    print(f"# Country n°{country} : {countries[country-1]}.")

    Names_numrow = {i:idx for i, idx in enumerate(Names.index)} # Keep initial row number
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
        for idx1, (Liste_idx, Liste_val) in enumerate(zip(Tfidf_idx, Tfidf_val)) :
            idx1 = Names_numrow[idx1]
            id1, lat1, lon1, cat1, cat_simpl1 = train['id'].iat[idx1], train['latitude'].iat[idx1], train['longitude'].iat[idx1], train['categories'].iat[idx1], train['category_simpl'].iat[idx1]
            for idx2, val in zip(Liste_idx, Liste_val) :
                if idx1 < idx2 :
                    id2, lat2, lon2, cat2, cat_simpl2 = train['id'].iat[idx2], train['latitude'].iat[idx2], train['longitude'].iat[idx2], train['categories'].iat[idx2], train['category_simpl'].iat[idx2]
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    dist = haversine(lat1,lon1,lat2,lon2)
                    same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='' and cat1 !='nan')
                    if key not in Cand and dist<=thr_distance and (same_cat or dist<=thr_distance_or_same_cat) and (same_cat or val >= thr_tfidf) :
                        poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
                        Cand[key] = int(poi1==poi2)
                        Added_p1.append([id1, poi1, int(poi1==poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1==poi2)

        p1 = p1.append(pd.DataFrame(Added_p1, columns=p1.columns)).reset_index(drop=True).copy()
        p2 = p2.append(pd.DataFrame(Added_p2, columns=p2.columns)).reset_index(drop=True).copy()
        print(f"Candidates added : {len(p1)-size1}/{len(p1)}.")
print('\n-> TF-IDF for contries finished.')
print(f"Candidates added : {len(p1)-size}.")



#################################################################################
## TF-IDF n°3b : for each countries (with few processed name)
#################################################################################
# Vincent

thr_tfidf_ = 0.45
thr_distance_ = 25
thr_distance_or_same_cat_ = 10

size = len(p1)

for country in [32] : #range(1, 30)
    
    # Reset parameter
    thr_tfidf = thr_tfidf_
    thr_distance = thr_distance_
    thr_distance_or_same_cat = thr_distance_or_same_cat_
    
    # Tune parameters for each country
    if country == 32 :
        thr_tfidf_ = 0.4
        thr_distance = 100 # no limit
        thr_distance_or_same_cat = 100 # no limit

    Names = train[train['country']==country]['name_initial_decode'].copy()
    if len(Names) == 0 : break
        
    print()
    print("#"*20)
    print(f"# Country n°{country} : {countries[country-1]}.")

    Names_numrow = {i:idx for i, idx in enumerate(Names.index)} # Keep initial row number
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
        for idx1, (Liste_idx, Liste_val) in enumerate(zip(Tfidf_idx, Tfidf_val)) :
            idx1 = Names_numrow[idx1]
            id1, lat1, lon1, cat1, cat_simpl1 = train['id'].iat[idx1], train['latitude'].iat[idx1], train['longitude'].iat[idx1], train['categories'].iat[idx1], train['category_simpl'].iat[idx1]
            for idx2, val in zip(Liste_idx, Liste_val) :
                if idx1 < idx2 :
                    id2, lat2, lon2, cat2, cat_simpl2 = train['id'].iat[idx2], train['latitude'].iat[idx2], train['longitude'].iat[idx2], train['categories'].iat[idx2], train['category_simpl'].iat[idx2]
                    key = f"{min(id1, id2)}-{max(id1, id2)}"
                    dist = haversine(lat1,lon1,lat2,lon2)
                    same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='' and cat1 !='nan')
                    if key not in Cand and dist<=thr_distance and (same_cat or dist<=thr_distance_or_same_cat) and (same_cat or val >= thr_tfidf) :
                        poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
                        Cand[key] = int(poi1==poi2)
                        Added_p1.append([id1, poi1, int(poi1==poi2)])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1==poi2)

        p1 = p1.append(pd.DataFrame(Added_p1, columns=p1.columns)).reset_index(drop=True).copy()
        p2 = p2.append(pd.DataFrame(Added_p2, columns=p2.columns)).reset_index(drop=True).copy()
        print(f"Candidates added : {len(p1)-size1}.")
print('\n-> TF-IDF for contries finished.')
print(f"Candidates added : {len(p1)-size}.")


#################################################################################
## Add candidates based on same name/phone/address
#################################################################################
# Vincent

def create_address(address, city) :
    if address == 'nan' or len(address) <= 1 :
        return 'nan'
    elif city == 'nan' or len(city) <= 1 :
        return 'nan'
    else : 
        return address.lower().strip() + '-' + city.lower().strip()
    
def find_potential_matchs(row) :
    name    = row['name']
    # Missing name
    if (name not in work_names) or name == 'nan' or len(name) <= 1 :
        return []
    
    phone   = row['phone']
    address_complet = create_address(row['address'], row['city'])

    # Non-missing name
    index = work_names[name].copy()

    if phone != 'nan' and len(phone)>1 and phone in work_phones :
        index += work_phones[phone]

    if address_complet != 'nan' and address_complet in work_address :
        index += work_address[address_complet]
        
        
    return list(set(index))

# =====================================================

# Création d'un df de travail
work = train.copy()

# Prepare columns
for c in ['name', 'address', 'city'] :
    work[c] = work[c].astype(str).str.lower()
work['index'] = work.index

work_names = work.groupby('name')['index'].apply(list).to_frame().reset_index()
work_names = dict(zip(work_names['name'], work_names['index']))
work_names = {name:Liste_idx for name, Liste_idx in work_names.items() if len(name)>=2 and len(Liste_idx)<=25} # Don't consider too widespread names

work_phones = work.groupby('phone')['index'].apply(list).to_frame().reset_index()
work_phones = dict(zip(work_phones['phone'], work_phones['index']))
work_phones = {phone:Liste_idx for phone, Liste_idx in work_phones.items() if len(phone)>=3 and len(Liste_idx)<=10} # Don't consider too widespread phone

work['address_complet'] = work.apply(lambda row : create_address(row['address'], row['city']), axis=1)
work_address = work.groupby('address_complet')['index'].apply(list).to_frame().reset_index()
work_address = dict(zip(work_address['address_complet'], work_address['index']))
work_address = {address:Liste_idx for address, Liste_idx in work_address.items() if len(address)>=3 and len(Liste_idx)<=10} # Don't consider too widespread address

# Process
#tqdm.pandas()
#Potential_on_NamePhone = work.progress_apply(find_potential_matchs, axis=1).to_list()
Potential_on_NamePhone = work.apply(lambda row : find_potential_matchs(row), axis=1).to_list()

# Don't keep pairs too far from each other
Potential_on_NamePhone_new = []

# Numpy for faster process
train_numpy = train[['name', 'latitude', 'longitude', 'category_simpl']].to_numpy()

# Filtre on dist
for i, Liste_idx in enumerate(Potential_on_NamePhone) :
    new = []
    name1, lat1, lon1, cat_simpl1  = train_numpy[i][0], train_numpy[i][1], train_numpy[i][2], train_numpy[i][3]
    for j, row in enumerate(train_numpy[Liste_idx]) :
        name2, lat2, lon2, cat_simpl2 = row[0], row[1], row[2], row[3]
        
        # if rare name, we are more tolerant
        if name1==name2 and len(work_names[name1]) <= 5 : 
            thr_distance = 100
        else :
            thr_distance = 26
            
        # if the category is usually far even for matchs
        if (cat_simpl1 in far_cat_simpl) or (cat_simpl2 in far_cat_simpl) :
            thr_distance = 350
            if (cat_simpl1 == 1) or (cat_simpl2 == 1) :
                thr_distance = 100000 # no limit
                
        # Add distance if long names (not a coincidence if they are equal)
        if name1==name2 and len(name1) >= 10 :
            thr_distance += 15
        
        # Process
        if haversine(lat1,lon1,lat2,lon2) > thr_distance :
            continue
        else :
            new.append(Liste_idx[j])
    Potential_on_NamePhone_new.append(new.copy())
    
Potential_on_NamePhone = Potential_on_NamePhone_new.copy()

del Potential_on_NamePhone_new, train_numpy
gc.collect()

# Number of potential matchs
print(f"Potential match on name/phone/address : {sum(len(x) for x in Potential_on_NamePhone)}.")

# Add matches
size1 = len(p1)
Added_p1, Added_p2 = [], []
try : seen
except : seen = set()
for idx1, Liste_idx in enumerate(Potential_on_NamePhone) :
    
    id1, lat1, lon1, cat1, cat_simpl1 = train['id'].iat[idx1], train['latitude'].iat[idx1], train['longitude'].iat[idx1], train['categories'].iat[idx1], train['category_simpl'].iat[idx1]
    for idx2 in Liste_idx :
        if idx1 != idx2 :
            id2, lat2, lon2, cat2, cat_simpl2 = train['id'].iat[idx2], train['latitude'].iat[idx2], train['longitude'].iat[idx2], train['categories'].iat[idx2], train['category_simpl'].iat[idx2]
            key = f"{min(id1, id2)}-{max(id1, id2)}"
            #same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='')
            if key not in Cand and key not in seen :
                seen.add(key)
                poi1, poi2 = train['point_of_interest'].iat[idx1], train['point_of_interest'].iat[idx2]
                Added_p1.append([id1, poi1, int(poi1==poi2)])
                Added_p2.append([id2, poi2])
                
p1 = p1.append(pd.DataFrame(Added_p1, columns=p1.columns)).reset_index(drop=True).copy()
p2 = p2.append(pd.DataFrame(Added_p2, columns=p2.columns)).reset_index(drop=True).copy()
print(f"Candidates added with name/phone/address similarity : {len(p1)-size1}.")

del Tfidf_idx, Tfidf_val, Cand, ID_to_POI, Potential_on_NamePhone, work, work_names, work_phones, work_address, Added_p1, Added_p2, seen, Names_numrow
gc.collect()
# Vincent
#################################################################################
#################################################################################
print()


# Reset index after Vincent's candidate addition
p1 = p1.reset_index(drop=True)
p2 = p2.reset_index(drop=True)



# remove duplicate pairs
p12 = pd.concat([p1['id'], p2['id']], axis=1)
p12.columns = ['id','id2']
p12 = p12.reset_index()

# flip - only keep one of the flipped pairs, the other one is truly redundant
idx = p12['id'] > p12['id2']
p12['t'] = p12['id']
p12['id'].loc[idx] = p12['id2'].loc[idx]
p12['id2'].loc[idx] = p12['t'].loc[idx]

p12 = p12.sort_values(by=['id', 'id2']).reset_index(drop=True)
p12 = p12.drop_duplicates(subset=['id', 'id2'])

# also drop id == id2 - it may happen
p12 = p12.loc[p12['id'] != p12['id2']]
p1 = p1.loc[p12['index']].reset_index(drop=True)
p2 = p2.loc[p12['index']].reset_index(drop=True)
del p12, idx
gc.collect()
y = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
print('removed duplicates', p1.shape[0], (p1['point_of_interest'] == p2['point_of_interest']).sum(), int(time.time() - start_time), 'sec')
# get stats
p1['y'] = y
a = p1.groupby('id')['y'].sum().reset_index()
print('count of all matching pairs is:', np.minimum(1,a['y']).sum()) 



# get CV*******************************************************************************************************************************
def cci(x, y):# count y in x (intersection)
    i = 0
    for j in range(1000):
        l1 = y.find(' ')
        if y[:l1+1] in x: # include space in search to avoid false positives (need to have trailing space in m_true)
            i += 1
        y = y[l1+1:]
        if ' ' not in y: # only 1 item left
            break
    return i

def get_CV(p1, p2, y, oof_preds, all_cuts = 1):
    cv0 = 0
    cut0 = 0
    # first, construct composite dataframe
    df2 = p1[['id']]
    df2['id2'] = p2['id']
    df2['y'] = y
    df2['match'] = oof_preds.astype('float32')
    df2 = df2.merge(train[['id', 'm_true']], on='id', how='left') # bring in m_true
    cut2 = 0.8 # hardcode for now
    ll = [0, .4, .45, .5, .55, .6, .65]
    if all_cuts == 0:
        ll = [0]
    for cut1 in ll:
        # select matching pairs
        cut = cut1
        if cut1 == 0: # true match, for max cv assessment only - this is CV if lgb model predicts a perfect answer
            matches = df2[['id', 'id2', 'match', 'y']].loc[df2['y']==1].reset_index(drop=True)
            matches['match'] = matches['y']
        else:
            matches = df2[['id', 'id2', 'match']].loc[df2['match']>cut].reset_index(drop=True) # call it a match if p > cut

        # construct POI from pairs
        poi_di = {} # maps each id to POI
        poi = 0
        id1, id2, preds = matches['id'].to_numpy(), matches['id2'].to_numpy(), matches['match'].to_numpy()
        for i in range(matches.shape[0]):
            i1 = id1[i]
            i2 = id2[i]
            if i1 in poi_di: # i1 is already in dict - assign i2 to the same poi
               poi_di[i2] = poi_di[i1]
            else:
                if i2 in poi_di: # i2 is already in dict - assign i1 to the same poi
                   poi_di[i1] =  poi_di[i2]
                else: # new poi, assign it to both
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
                if poi_di[i2] != poi_di[i1] and pred > cut2: # 2 different groups - put them all in lower one
                    m = min(poi_di[i2], poi_di[i1])
                    poi_di[i1] = m
                    poi_di[i2] = m
                    j += 1
            if j == 0:
                break
        
        # construct list of id/poi pairs
        m2 = pd.DataFrame(matches['id'].append(matches['id2'], ignore_index=True))
        m2 = m2.drop_duplicates()
        m2['poi'] = m2[0].map(poi_di)

        # predicted true groups
        m2     = m2.sort_values(by=['poi', 0]).reset_index(drop=True)
        ids, pois = m2[0].to_numpy(), m2['poi'].to_numpy()
        poi0   = pois[0]
        id0    = ids[0]
        di_poi = {} # this maps POI to list of all ids that belong to it
        for i in range(1, m2.shape[0]):
            if pois[i] == poi0: # this id belongs to the same POI as prev id - add it to the list
                id0 = str(id0) + ' ' + str(ids[i]) # id0 is list of all ids that belong to current POI
            else:
                di_poi[poi0]    = str(id0) + ' ' # need to have trailing space in m_true
                poi0            = pois[i]
                id0             = ids[i]
        di_poi[poi0] = str(id0) + ' ' # need to have trailing space in m_true
        m2['m2'] = m2['poi'].map(di_poi) # this is the list of all matches
        m2.columns = ['id', 'poi', 'm2']

        # bring predictions into final result
        p1_tr = train[['id', 'm_true']].merge(m2[['id', 'm2']], on='id', how='left')
        p1_tr['m2'] = p1_tr['m2'].fillna('missing')
        idx = p1_tr['m2'] == 'missing'
        p1_tr['m2'].loc[idx] = p1_tr['id'].astype('str').loc[idx] + ' ' # fill missing values with id - those correspond to 1 id per poi

        # compare to true groups
        ii = np.zeros(p1_tr.shape[0], dtype=np.int32)
        x1, x2 = p1_tr['m_true'].to_numpy(), p1_tr['m2'].to_numpy()
        for i in range(p1_tr.shape[0]): ii[i] = cci(x1[i], x2[i])
        p1_tr['intersection'] = ii
        p1_tr['len_pred'] = p1_tr['m2'].apply(lambda x: x.count(' '))
        p1_tr['len_true'] = p1_tr['m_true'].apply(lambda x: x.count(' '))
        p1_tr['union'] = p1_tr['len_true'] + p1_tr['len_pred'] - p1_tr['intersection']
        cv = (p1_tr['intersection'] / p1_tr['union']).mean()
        if cv > cv0 or cut0 == 0: # always overwrite 0, that was only a max assessment
            cv0 = cv
            cut0 = cut1
        print(cut1, 'CV***', np.round(cv, 4), int(time.time() - start_time), 'sec')
    print('best cut is', cut0, 'best CV is', np.round(cv0, 4), '*************************************************************************')
    return cut0, cv0


# shuffle p1/p2
idx = np.random.permutation(p1.index)
p2 = p2.loc[idx].reset_index(drop=True)
p1 = p1.loc[idx].reset_index(drop=True)
y = np.array(p1['point_of_interest'] == p2['point_of_interest']).astype(np.int8)
del idx

# add other columns - needed for FE
cols = ['id', 'name', 'latitude', 'longitude', 'address', 'country', 'url', 'phone', 'city', 'categories', 'category_simpl', 'categories_split', 'cat2']
p1 = p1[['id']].merge(train[cols], on='id', how='left')
p2 = p2[['id']].merge(train[cols], on='id', how='left')

# check for flipped sign on longitude - this may help test data a lot; test it? Move this code up to apply to "train"
dist = distance(np.array(p1['latitude']), np.array(p1['longitude']), np.array(p2['latitude']), np.array(p2['longitude']))
df = pd.DataFrame(dist)
df.columns = ['dist']
df['dist'] = df['dist'].astype('int32')
df['dist1'] = (111173.444444444 * np.abs(p1['latitude'] - p2['latitude'])).astype('int32')
df['dist2'] = np.sqrt(np.maximum(0, (1.0*df['dist'])**2 - df['dist1']**2)).astype('int32')
idx = ((df['dist1'] < 10000) & (df['dist2'] > 1000000) & (np.abs(p1['longitude']+p2['longitude']) < .1)) & (p1['country']==p2['country'])
# this selects only 3 cases in train data, but possibly more in test, so keep it becasue it is basically free
print('flipped sign of longitude for', idx.sum(), 'points')
p1['longitude'].loc[idx] *= -1 # flip(correct) sign
del df, idx, dist, a
gc.collect()


# FE1***************************************************************************************************************************************
def FE1(p1, p2):
    print('FE: start', int(time.time() - start_time), 'sec')
    dist = distance(np.array(p1['latitude']), np.array(p1['longitude']), np.array(p2['latitude']), np.array(p2['longitude']))
    df = pd.DataFrame(dist)
    df.columns = ['dist']
    df['dist'] = df['dist'].astype('int32')
    df['dist1'] = (111173.444444444 * np.abs(p1['latitude'] - p2['latitude'])).astype('int32') # now on the same scale as dist
    df['dist2'] = np.sqrt(np.maximum(0, (1.0*df['dist'])**2 - df['dist1']**2)).astype('int32') # get this by subtraction
    for col in ['dist', 'dist1', 'dist2']:
        df[col] = np.exp(np.round(np.log(1+df[col]), 1)-0.5).astype('int32')
        df[col] = np.minimum(100000, np.round(df[col], -1))
    del dist
    gc.collect()

    # country - categorical
    df['country'] = p1['country']
    df['country'].loc[p1['country'] != p2['country']] = 0
    df['country'] = df['country'].astype('category')

    # cat2 - categorical
    df['cat2a'] = np.minimum(p1['cat2'], p2['cat2']).astype('category')
    df['cat2b'] = np.maximum(p1['cat2'], p2['cat2']).astype('category')

    ii = np.zeros(df.shape[0], dtype=np.int16)      # integer placeholder
    num_digits2 = 2 # digits for ratios
    for col in ['name', 'categories', 'address']:
        print('FE: start processing column', col, int(time.time() - start_time), 'sec')
        x1, x2 = p1[col].to_numpy(), p2[col].to_numpy()

        # pi1 = partial intersection, start of string
        for i in range(df.shape[0]): ii[i] = pi(x1[i], x2[i])
        df[col+'_pi1'] = ii

        # lcs2 = longest common substring
        for i in range(df.shape[0]): ii[i] = lcs2(x1[i], x2[i])
        df[col+'_lcs2'] = ii

        # lcs = longest common subsequence
        for i in range(df.shape[0]): ii[i] = lcs(x1[i], x2[i])
        df[col+'_lcs'] = ii

        # ll1 - min length of this column
        ll1 = np.maximum(1, np.minimum(p1[col].apply(len), p2[col].apply(len))).astype(np.int8) # min length
        ll2 = np.maximum(1, np.maximum(p1[col].apply(len), p2[col].apply(len))).astype(np.int8) # max length
        
        # compound features (ratios) ****************
        # pi1 / ll1 = r1
        df[col+'_pi1_r1'] = np.round((df[col+'_pi1'] / ll1), num_digits2).astype('float32')
        
        # lcs2 / ll1 = r1
        df[col+'_lcs2_r1'] = np.round((df[col+'_lcs2'] / ll1), num_digits2).astype('float32')

        # lcs2 / ll2 = r2
        df[col+'_lcs2_r2'] = np.round((df[col+'_lcs2'] / ll2), num_digits2).astype('float32')

        # lcs / ll1 = r1
        df[col+'_lcs_r1'] = np.round((df[col+'_lcs'] / ll1), num_digits2).astype('float32')

        # lcs / ll2 = r2
        df[col+'_lcs_r2'] = np.round((df[col+'_lcs'] / ll2), num_digits2).astype('float32')

        # ll1 / ll2 = r3
        df[col+'_r3'] = np.round(ll1 / ll2, num_digits2).astype('float32')

        # lcs2 / lcs = r4
        df[col+'_lcs_r4'] = np.round((df[col+'_lcs2'] / np.maximum(1, df[col+'_lcs'])), num_digits2).astype('float32')
    del x1, x2, ii, ll1, ll2
    gc.collect()

    # NA count for some text columns
    for col in ['city', 'address']:
        df[col+'_NA'] = ((p1[col] == 'nan') * 1 + (p2[col] == 'nan') * 1 + (p1[col] == '') * 1 + (p2[col] == '') * 1).astype('int8')
        print('NA - finished', col, int(time.time() - start_time), 'sec')

    # match for some text columns
    df['phone_m10'] = ((p1['phone'] == p2['phone']) & (p1['phone'] != '') & (p1['phone'] != 'nan')).astype('int8')
    df['url_m5'] = ((p1['url'].str[:5] == p2['url'].str[:5]) & (p1['url'] != '') & (p1['url'] != 'nan')).astype('int8')

    # simp cat match
    mask1 = (p1['category_simpl']>=1) & (p1['category_simpl'] == p2['category_simpl'])
    mask2 = (p1['category_simpl']==0) & (p1['categories'] == p2['categories'])
    df['same_cat_simpl'] = ((mask1) | (mask2)).astype('int8')
    del mask1, mask2
    gc.collect()

    # ratio of dist to mean dist by cat2
    df['dm'] = df['cat2a'].astype('int32').map(dist_by_cat2)
    df['dist_r1'] = np.minimum(800, np.exp(np.round(np.log(1+df['dist'] / df['dm']), 1)) - 1).astype('float32') # median: log scale, cap at 800

    # ratio of dist to mean dist by category_simpl
    df['cat_simpl'] = p1['category_simpl'].astype('int16')
    df['cat_simpl'].iloc[df['same_cat_simpl'] == 0] = 0 # not a match - make it 0
    df['cat_simpl'] = df['cat_simpl'].astype('category')
    df['dm'] = df['cat_simpl'].astype('int32').map(dist_by_category_simpl)
    df['dist_r2'] = np.minimum(800, np.exp(np.round(np.log(1+df['dist'] / df['dm']), 1)) - 1).astype('float32') # median: log scale, cap at 800
    df.drop(['dm', 'cat_simpl'],axis=1, inplace=True)
    print('finished ratio of dist to mean dist by category_simpl', int(time.time() - start_time), 'sec')
    p1 = p1[['id','name']]
    p2 = p2[['id','name']]
    gc.collect()

    # number of times col appears in this data
    cc_cap = 10000 # cap on counts
    for col in ['id', 'name']:
        p12 = p1[[col]].append(p2[[col]], ignore_index=True) # count it in both p1 and p2
        df1 = p12[col].value_counts()
        p1['cc'] = p1[col].map(df1)
        p2['cc'] = p2[col].map(df1)
        del p12, df1
        gc.collect()
        # features
        df[col+'_cc_min'] = np.minimum(cc_cap, np.minimum(p1['cc'], p2['cc'])) # min, capped at X, scaled
        df[col+'_cc_max'] = np.minimum(cc_cap, np.maximum(p1['cc'], p2['cc'])) # max, capped at X, scaled
        # log-transform
        df[col+'_cc_min'] = (np.exp(np.round(np.log(df[col+'_cc_min']+1), 1))-.5).astype('int16')
        df[col+'_cc_max'] = (np.exp(np.round(np.log(df[col+'_cc_max']+1), 1))-.5).astype('int16')
        p1.drop('cc', axis=1, inplace=True)
        p2.drop('cc', axis=1, inplace=True)
    print('finished min/max cc', int(time.time() - start_time), 'sec')
    return df
# end of FE1***************************************************************************************************************************************

df = FE1(p1, p2) # call FE1

# drop unneeded cols to save RAM - only keep id, reload the rest later
p1 = p1[['id']]
p2 = p2[['id']]
gc.collect()

features = df.columns
types = df.dtypes
df = np.array(df, dtype=np.float32) # turn into np array to avoid memory spike


# print size of large vars
for i in dir():
    ss = sys.getsizeof(eval(i))
    if ss > 1000000:
        print(i, np.round(ss/1000000,1 ))

# lgb1 - fast, just to reduce data size
print('data size is', df.shape)
folds = 5 # number of folds
oof_preds = np.zeros(df.shape[0], dtype=np.float32)
#kf = KFold(n_splits=folds, shuffle=False)#, random_state=13)
#kf = GroupKFold(n_splits=folds)
kf = StratifiedKFold(n_splits=folds)
fi = np.zeros(df.shape[1])
#g = np.array(p1[['id']].merge(train[['id','point_of_interest']], on='id', how='left')['point_of_interest'])
for fold, (train_idx, valid_idx) in enumerate(kf.split(df, y)):
#for fold, (train_idx, valid_idx) in enumerate(kf.split(df, y+2*np.minimum(15,df[:,6])+2*100*np.minimum(11,df[:,3]))):
    x_train, x_valid = df[train_idx], df[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=[3, 4, 5]) # country, cat2a, cat2b
    lgb_valid = lgb.Dataset(x_valid, y_valid, categorical_feature=[3, 4, 5])
    params = {'seed':13, 'objective':'binary', 'verbose':-1, 'num_leaves':255, 'learning_rate':0.05}
    model = lgb.train(params, lgb_train, 20000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=30, verbose_eval=20000)
    oof_preds[valid_idx] = model.predict(x_valid).ravel()
    fi += model.feature_importance(importance_type='gain', iteration=model.best_iteration)
    m = model.save_model('model_a'+str(fold)+'.txt') # save model
del x_train, x_valid, train_idx, valid_idx, y_train, y_valid
gc.collect()
# fi
fi = fi / fi.sum()
feature_imp = pd.DataFrame(zip(fi, features, types), columns=['Value','Feature', 'Type'])
feature_imp = feature_imp.sort_values(by='Value').reset_index(drop=True)
feature_imp['Value'] = np.round(feature_imp['Value'], 5)
print( feature_imp[:10] )  # worst
print( feature_imp[-10:] ) # best
get_CV(p1, p2, y, oof_preds) # get CV



# check several cut levels
df2 = pd.DataFrame(y)
df2.columns = ['y']
df2['p'] = oof_preds
df2['id'] = p1['id']
y0 = df2['y'].sum()
a0 = np.minimum(1,df2.groupby('id')['y'].sum()).sum()
for cut in [-.1, .001, .002,  .003, .005, .007, .01, .02, .05]:
    a = df2.loc[df2['p']>cut].groupby('id')['y'].sum().reset_index()
    print('reduce data size:', cut, df2.loc[df2['p']>cut].shape[0], df2['y'].loc[df2['p']>cut].sum(), y0 - df2['y'].loc[df2['p']>cut].sum(), np.minimum(1,a['y']).sum(), a0- np.minimum(1,a['y']).sum())

# select subset of data
idx = oof_preds > .007 # hardcode .007 here - for now.
p1 = p1.loc[idx].reset_index(drop=True)
p2 = p2.loc[idx].reset_index(drop=True)
df = df[idx]
y  = y[idx]

df = pd.DataFrame(df) # convert back to dataframe
df.columns = features
for col in types.index:
    df[col] = df[col].astype(types[col])
del df2
gc.collect()







# now that i have a smaller dataset, expand features
# FE2***************************************************************************************************************************************
def FE2(df, p1, p2):
    cat_links = pd.read_pickle(path+'link_between_categories.pkl') # link-between-categories
    Dist_quantiles = pd.read_pickle(path+'Dist_quantiles_per_catpairs.pkl')  # dist-quantiles-per-cat

    cat_links_ratio, cat_links_ratio_all = [], []
    dist_qtl = []


    p1 = p1[['id']].merge(train[['id','categories_split']], on='id', how='left')
    p2 = p2[['id']].merge(train[['id','categories_split']], on='id', how='left')
    for i, (L1, L2) in enumerate(zip(p1['categories_split'], p2['categories_split'])) :
        # Find the biggest score corresponding to one of the possible category-pairs
        s0, s1 = 0, 0
        q = [Dist_quantiles[('', '')].copy()] # default : couple of nan
        for cat1 in L1 :
            for cat2 in L2 :
                key = tuple(sorted([cat1, cat2]))
                if key in cat_links :
                    x = cat_links[key]
                    if x[0] > s0 : s0 = x[0]
                    if x[1] > s1 : s1 = x[1]
                if key in Dist_quantiles :
                    q.append(Dist_quantiles[key])
        # Append
        cat_links_ratio.append(s0)
        cat_links_ratio_all.append(s1)
        dist_qtl.append(np.max(np.array(q), axis=0))

    # Drop useless column
    #train.drop(columns = ["categories_split"], inplace=True)
    del Dist_quantiles, cat_links
    gc.collect()

    # add other columns - needed for FE
    p1 = p1[['id']].merge(train, on='id', how='left')
    p2 = p2[['id']].merge(train, on='id', how='left')


    # Raise recursion limit
    import resource, sys
    resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**8)

    # 1. Simply connected components
    from collections import defaultdict
    connected_components = defaultdict(set)

    def dfs(res, node):
        #global connected_components
        if node not in connected_components:
            connected_components[node] = set()
            for next_ in res[node]:
                dfs(res, next_)
                connected_components[node] = connected_components[next_]
            connected_components[node].add(node)

    # --------------------------------------
    # Create graph
    from collections import defaultdict
    graph = defaultdict(set)
    for id1, id2 in zip(p1['id'], p2['id']) :
        if id1 != id2 :
            graph[id1].add(id2)
            graph[id2].add(id1)
            
    # Find connected comoponents
    for node_ in graph:
        dfs(graph, node_)
    connected_components = map(tuple, connected_components.values())
    unique_components = set(connected_components)
    Connexes =  [list(x) for x in unique_components]
    Len_connect = {}
    for Liste_index in Connexes :
        for idx in Liste_index :
            Len_connect[idx] = min(len(Liste_index), 200)
            
    # Add feature
    df["Nb_connect1"] = p1["id"].apply(lambda idx : Len_connect[idx]).astype('int16')
    df["Nb_connect2"] = p2["id"].apply(lambda idx : Len_connect[idx]).astype('int16')

    del Connexes, connected_components, Len_connect
    gc.collect()


    # 2. Strongly connected components
    def get_strongly_connected_components(graph):
        seen = set()
        components = []
        for node in graph:
            if node not in seen:
                component = []
                nodes = {node}
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    component.append(node)
                    nodes.update(graph[node].difference(seen))
                components.append(component)
        return components

    # Strongly connected components
    Connexes = get_strongly_connected_components(graph)

    Len_strong_connect = {}
    for Liste_index in Connexes :
        for idx in Liste_index :
            Len_strong_connect[idx] = min(len(Liste_index), 200)
            
    # Add feature
    df["Nb_strong_connect"] = p1["id"].apply(lambda idx : Len_strong_connect[idx]).astype('int16')

    del Connexes, Len_strong_connect, graph
    gc.collect()



    ################################
    # Vincent
    df['cat_link_score'] = cat_links_ratio.copy()
    df['cat_link_score_all'] = cat_links_ratio_all.copy()
    # Cat link scores
    for x in ['cat_link_score', 'cat_link_score_all'] :
        df[x] = np.round(df[x], 3).astype('float32')
    df.loc[:, [x+'_pair' for x in col_cat_distscores]] = dist_qtl.copy()
    for x in col_cat_distscores:
        df[x+'_pair'] = df[x+'_pair'].astype('float32')
    del cat_links_ratio, cat_links_ratio_all, dist_qtl
    gc.collect()
    print('finished cat_link_score_all', int(time.time() - start_time), 'sec')

    # Add features
    for x in col_cat_distscores + ['cat_solo_score', 'freq_pairing_with_other_groupedcat'] :
        df[x+'_1'] = np.maximum(p1[x], p2[x]).astype('float32')
        df[x+'_2'] = np.minimum(p1[x], p2[x]).astype('float32')
    print('finished freq_pairing_with_other_groupedcat', int(time.time() - start_time), 'sec')

    # Feature engineering : dist divided by previous features
    MAX, eps = 1e6, 1e-6
    for x in col_cat_distscores[1:] :
        #############################
        df[x+'_ratiodist_1'] = df['dist'] / (df[x+'_1'] + eps) # epsilon to avoid dividing by 0
        df[x+'_ratiodist_2'] = df['dist'] / (df[x+'_2'] + eps) # epsilon to avoid dividing by 0
        df[x+'_ratiodist_pair'] = df['dist'] / (df[x+'_pair'] + eps) # epsilon to avoid dividing by 0
        # Avoid too high values
        df[x+'_ratiodist_1'] = df[x+'_ratiodist_1'].apply(lambda x : x if x <= MAX else MAX).astype('float32')
        df[x+'_ratiodist_2'] = df[x+'_ratiodist_2'].apply(lambda x : x if x <= MAX else MAX).astype('float32')
        df[x+'_ratiodist_pair'] = df[x+'_ratiodist_pair'].apply(lambda x : x if x <= MAX else MAX).astype('float32')
    # Useless after all (low fi; it's better to only keep dist ratio)
    df.drop(columns = [x+'_pair' for x in col_cat_distscores], inplace=True)
    print('finished dist divided by previous features', int(time.time() - start_time), 'sec')

    # Link between grouped categories
    cat_links = pd.read_pickle(path+'link_between_grouped_categories.pkl')
    L1, L2 = [], []
    for cat1, cat2 in zip(p1['category_simpl'], p2['category_simpl']) :
        key = tuple(sorted([cat1, cat2]))
        x = [0, 0]
        if key in cat_links :
            x = cat_links[key]
        L1.append(x[0])
        L2.append(x[1])
    df['grouped_cat_link_score'] = L1.copy()
    df['grouped_cat_link_score_all'] = L2.copy()
    for x in ['grouped_cat_link_score', 'grouped_cat_link_score_all'] :
        df[x] = np.round(df[x], 3).astype('float32')
    print('finished Link between grouped categories', int(time.time() - start_time), 'sec')
    del L1, L2, cat_links
    gc.collect()
    ################################
    drops = ['freq_pairing_with_other_groupedcat', 'cat_solo_score', 'Nb_multiPoi', 'mean', 'q25', 'q50', 'q75', 'q90', 'q99']
    p1.drop(drops, axis=1, inplace=True)
    p2.drop(drops, axis=1, inplace=True)



    ii  = np.zeros(df.shape[0], dtype=np.int16)      # integer placeholder
    fi  = np.zeros(df.shape[0], dtype=np.float32)    # floating point placeholder
    fi2 = np.zeros(df.shape[0], dtype=np.float32)    # floating point placeholder2
    num_digits  = 3 # digits for floats
    num_digits2 = 2 # digits for ratios
    # list of features to skip due to low fi (<0.00015)
    f_skip0 = set(['phone_dsm2','state_dsm2','city_dsm2','address_dsm2','nameC_dsm2',
    'state_ll1','city_lcs_r4','zip_lllcs_r1','zip_pi2_r1','phone_ld','phone_NA','state_dsm2','city_lcs2','city_pi2_r1','state_dsm1','name_initial_m5','categories_lllcs','state_lcs2_r2','address_lllcs_r2','address_lllcs','address_lcs2','city_pi2','url_ljw','city_lcs_r2','word_n_m_cc','phone_dsm2','address_lcs','word_c_22_m','city_lcs_r1','address_lcs_r4',
    'name_initial_decode_m10','phone_r3','word_c_m_cc','url_lcs2_r2','city_lllcs','url_NA','zip_lcs','state_lcs2','state_pi1','name_initial_m10','url_lcs2','zip_r3','zip_ld','city_pi1_r1','name_m10','zip_ll1','state_lcs','phone_pi1','url_dsm2',
    'state_pi2_r1','state_pi1_r1','state_pi2','url_lllcs','name_initial_NA','name_initial_decode_NA','categories_NA','url_pi2','url_pi1','zip_lcs_r4','categories_m10','state_lllcs_r2','zip_lllcs','state_lllcs','zip_m5','categories_m5','nameC_lcs_r4','nameC_m5','nameC_cclcs','nameC_m10','nameC_NA','categories_cc_max','url_lcs_r4','address_cclcs','phone_lcs_r4','state_cc_max','phone_lllcs_r2','phone_lllcs_r1','phone_lcs_r2','phone_lcs_r1',
    'phone_lcs2_r2','phone_lcs2_r1','phone_pi2_r1','phone_pi1_r1','phone_m5','name_NA','phone_cclcs','state_m10','state_m5','city_m10','url_m5','city_cclcs',
    'url_m10','state_cclcs','url_cclcs','zip_m10','categories_cclcs','zip_dsm2','state_lllcs_r1','max_categories_cc','zip_cclcs','url_lcs2_r1','address_m10','url_lllcs_r1','url_lllcs_r2','city_m5','city_lllcs_r2','url_pi2_r1','phone_lllcs','zip_pi1',
    'zip_lcs2_r1','url_pi1_r1','zip_dsm1','zip_lllcs_r2','zip_lcs_r2','phone_ll1','url_lcs_r1','phone_lcs','zip_lcs2_r2','address_m5','city_lllcs_r1','zip_lcs2'])
    f_skip1 = set(df.columns)
    for col in ['name_initial', 'name_initial_decode', 'nameC', 'name', 'categories', 'address', 'url', 'city', 'state', 'zip', 'phone']:
        print('FE2: start processing column', col, int(time.time() - start_time), 'sec')
        x1, x2 = p1[col].to_numpy(), p2[col].to_numpy()

        name = col+'_cclcs' # cclcs = count of common substrings of length X+
        if name not in f_skip1 and  name not in f_skip0: # skip to save time
            for i in range(df.shape[0]): ii[i] = cc_lcs(x1[i], x2[i], 4)
            df[name] = ii
            
        name = col+'_lllcs' # lllcs = total length of common substrings of length X+
        if name not in f_skip1: # skip to save time
            min_len = 5
            if col == 'nameC':
                min_len = 1
            for i in range(df.shape[0]): ii[i] = ll_lcs(x1[i], x2[i], min_len)
            df[name] = ii

        name = col+'_lcs2' # lcs2 = longest common substring
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): ii[i] = lcs2(x1[i], x2[i])
            df[name] = ii

        name = col+'_lcs' # lcs = longest common subsequence
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): ii[i] = lcs(x1[i], x2[i])
            df[name] = ii

        name = col+'_pi1' # pi1 = partial intersection, start of string
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): ii[i] = pi(x1[i], x2[i])
            df[name] = ii

        name = col+'_pi2' # pi2 = partial intersection, end of string
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): ii[i] = pi2(x1[i], x2[i])
            df[name] = ii

        name = col+'_ld' # ld = Levenshtein.distance
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): ii[i] = Levenshtein.distance(x1[i], x2[i])
            df[name] = ii

        name = col+'_ljw' # ljw = Levenshtein.jaro_winkler (float)
        if name not in f_skip1: # skip to save time
            for i in range(df.shape[0]): fi[i] = Levenshtein.jaro_winkler(x1[i], x2[i])
            df[name] = np.round(fi, num_digits).astype(np.float32) # round

        # dsm = difflib.SequenceMatcher (float); not symmetrical, do apply twice!
        for i in range(df.shape[0]): fi[i] = difflib.SequenceMatcher(None, x1[i], x2[i]).ratio()
        for i in range(df.shape[0]): fi2[i] = difflib.SequenceMatcher(None, x2[i], x1[i]).ratio()
        df[col+'_dsm1'] = np.round(np.minimum(fi, fi2), num_digits).astype(np.float32) # round
        df[col+'_dsm2'] = np.round(np.maximum(fi, fi2), num_digits).astype(np.float32) # round

        # ll1 - min length of this column
        ll1 = np.maximum(1, np.minimum(p1[col].apply(len), p2[col].apply(len))).astype(np.int8) # min length
        ll2 = np.maximum(1, np.maximum(p1[col].apply(len), p2[col].apply(len))).astype(np.int8) # max length
        df[col+'_ll1'] = ll1
        
        # compound features (ratios) ****************
        # pi1 / ll1 = r1
        df[col+'_pi1_r1'] = np.round((df[col+'_pi1'] / ll1), num_digits2).astype('float32')
        
        # pi2 / ll1 = r1
        df[col+'_pi2_r1'] = np.round((df[col+'_pi2'] / ll1), num_digits2).astype('float32')

        # lcs2 / ll1 = r1
        df[col+'_lcs2_r1'] = np.round((df[col+'_lcs2'] / ll1), num_digits2).astype('float32')

        # lcs2 / ll2 = r2
        df[col+'_lcs2_r2'] = np.round((df[col+'_lcs2'] / ll2), num_digits2).astype('float32')

        # lcs / ll1 = r1
        df[col+'_lcs_r1'] = np.round((df[col+'_lcs'] / ll1), num_digits2).astype('float32')

        # lcs / ll2 = r2
        df[col+'_lcs_r2'] = np.round((df[col+'_lcs'] / ll2), num_digits2).astype('float32')

        # lllcs / ll1 = r1
        df[col+'_lllcs_r1'] = np.round((df[col+'_lllcs'] / ll1), num_digits2).astype('float32')

        # lllcs / ll2 = r2
        df[col+'_lllcs_r2'] = np.round((df[col+'_lllcs'] / ll2), num_digits2).astype('float32')

        # ll1 / ll2 = r3
        df[col+'_r3'] = np.round(ll1 / ll2, num_digits2).astype('float32')

        # lcs2 / lcs = r4
        df[col+'_lcs_r4'] = np.round((df[col+'_lcs2'] / np.maximum(1, df[col+'_lcs'])), num_digits2).astype('float32')

        # count of NAs
        df[col+'_NA'] = ((p1[col] == 'nan') * 1 + (p2[col] == 'nan') * 1 + (p1[col] == '') * 1 + (p2[col] == '') * 1).astype('int8')
        
        # match5 - if not NA
        df[col+'_m5'] = ((p1[col].str[:5] == p2[col].str[:5]) & (p1[col] != '') & (p2[col] != '') & (p1[col] != 'nan') & (p2[col] != 'nan')).astype('int8')

        # match10 - if not NA
        df[col+'_m10'] = ((p1[col].str[:10] == p2[col].str[:10]) & (p1[col] != '') & (p2[col] != '') & (p1[col] != 'nan') & (p2[col] != 'nan')).astype('int8')
    # drop skipped columns
    df.drop(list(f_skip0.intersection(df.columns)), axis=1, inplace=True)
    del x1, x2, fi, fi2, ll1, ll2
    gc.collect()



    # cap features - after ratios. To reduce overfitting (cap determined to impact <0.1% of cases)
    cap_di = {'address_cclcs':5,'address_lcs':95,'address_lcs2':94,'address_ld':69,'address_ll1':100,'address_lllcs':94,'address_pi1':35,'address_pi2':94,'categories_lcs':40,'categories_lcs2':38,'categories_ld':58,'categories_ll1':46,'categories_lllcs':41,'categories_pi1':38,'categories_pi2':38,'city_lcs':19,'city_lcs2':19,'city_ld':25,'city_ll1':19,'city_lllcs':19
              ,'city_pi1':19,'city_pi2':19,'name_cclcs':4,'name_lcs':48,'name_lcs2':39,'name_ld':49,'name_ll1':58,'name_lllcs':44,'name_pi1':38,'name_pi2':37
              ,'phone_lcs2':10,'phone_ld':10,'phone_pi1':10,'phone_pi2':10,'state_lcs':20,'state_lcs2':20,'state_ld':27,'state_ll1':21,'state_lllcs':20,'state_pi1':20,'state_pi2':20,'url_lcs':63,'url_lcs2':47,'url_ld':105,'url_ll1':76,'url_lllcs':59,'url_pi1':46,'url_pi2':41,'zip_lcs':9,'zip_ld':9,'zip_ll1':9,'zip_lllcs':9,'zip_pi2':9}
    for col in cap_di.keys():
        if col in df.columns:
            df[col] = np.minimum(df[col], cap_di[col])

    # do something to reduce cardinality of ratios
    # round some floats to 2 digits, not 3
    for col in ['name_initial_ljw', 'name_initial_decode_ljw', 'name_initial_dsm1', 'name_initial_dsm2', 'name_initial_decode_dsm1', 'name_initial_decode_dsm2', 'zip_ljw', 'address_dsm1', 'address_dsm2', 'address_lcs_r4', 'address_ljw', 'categories_dsm1', 'categories_dsm2'
                , 'categories_ljw', 'city_dsm1', 'city_dsm2', 'city_ljw', 'name_dsm1', 'name_dsm2', 'name_ljw', 'nameC_dsm1', 'nameC_dsm2', 'nameC_ljw', 'state_dsm1', 'state_dsm2', 'state_ljw', 'url_dsm1', 'url_dsm2', 'url_ljw']:
        if col in df.columns:
            df[col] = np.round(df[col], 2)

    # log-transform/round some features to reduce cardinality
    for col in ['q75_ratiodist_pair','q90_ratiodist_pair','mean_ratiodist_pair','q50_ratiodist_pair','q25_ratiodist_pair','q99_ratiodist_pair','q50_ratiodist_2', 'q90_ratiodist_2', 'q25_ratiodist_2', 'q25_ratiodist_1', 'q75_ratiodist_2', 'mean_ratiodist_2', 'q99_ratiodist_2', 'q75_ratiodist_1', 'q50_ratiodist_1', 'q90_ratiodist_1', 'q99_ratiodist_1', 'mean_ratiodist_1']:
        if col in df.columns:
            df[col] = np.exp(np.round(np.log(df[col]+1), 1))


    # number of times col appears in this data
    cc_cap = 10000 # cap on counts
    for col in ['name', 'address', 'categories', 'id', 'city', 'state', 'zip', 'phone']:
        p12 = p1[[col]].append(p2[[col]], ignore_index=True) # count it in both p1 and p2
        df1 = p12[col].value_counts()
        p1['cc'] = p1[col].map(df1)
        p2['cc'] = p2[col].map(df1)
        del p12, df1
        gc.collect()
        # features
        df[col+'_cc_min'] = np.minimum(cc_cap, np.minimum(p1['cc'], p2['cc'])) # min, capped at X, scaled
        df[col+'_cc_max'] = np.minimum(cc_cap, np.maximum(p1['cc'], p2['cc'])) # max, capped at X, scaled
        # log-transform
        df[col+'_cc_min'] = (np.exp(np.round(np.log(df[col+'_cc_min']+1), 1))-.5).astype('int16')
        df[col+'_cc_max'] = (np.exp(np.round(np.log(df[col+'_cc_max']+1), 1))-.5).astype('int16')
        p1.drop('cc', axis=1, inplace=True)
        p2.drop('cc', axis=1, inplace=True)
    print('finished min/max cc', int(time.time() - start_time), 'sec')
    # drop unneeded features to same memory
    p1.drop(['url', 'city', 'state', 'zip', 'phone', 'country'], axis=1, inplace=True)
    p2.drop(['url', 'city', 'state', 'zip', 'phone', 'country'], axis=1, inplace=True)



    # find words in categories
    words_c = ['academic', 'airport', 'apartment', 'art', 'atm', 'auto', 'baker', 'bank', 'bar', 'barber', 'beach', 'building', 'burger', 'bus', 'cafe', 'car', 'center', 'church', 'classroom', 'cloth', 'club', 'coffee', 'college', 'concert', 'condo', 'construction', 'convenience', 'dentist', 'department', 'development', 'doctor', 'electronic', 'entertainment', 'event', 'factor', 'fast', 'field', 'food', 'furniture', 'gas', 'grocer', 'gym', 'hall', 'hospital', 'hotel', 'hous', 'joint', 'legal', 'mall', 'market', 'medic', 'metro', 'miscellan', 'mobile', 'movie', 'museum', 'music', 'office', 'park', 'pharmac', 'phone', 'pizza', 'place', 'play', 'plaza', 'pool', 'post', 'rental', 'residential', 'resort', 'restaurant', 'room', 'salon', 'school', 'service', 'shoe', 'shop', 'space', 'sport', 'stadium', 'station', 'stop', 'store', 'student', 'theater', 'train', 'travel', 'universit', 'video', 'work']
    w1 = np.zeros([p1.shape[0], len(words_c)], dtype=np.int8)
    w2 = np.zeros([p2.shape[0], len(words_c)], dtype=np.int8)
    for i, word in enumerate(words_c): # currently 90
        w1[p1['categories'].str.contains(word, regex=False), i] = 1
        w2[p2['categories'].str.contains(word, regex=False), i] = 1
        # features: match for each word. Only for words with high fi. Word order matters - do not change it.
        if i in [22]: # college
            df['word_c_'+str(i)+'_m'] = (w1[:,i] * w2[:,i]).astype('int8')
    print(i+1, 'finished words in categories', int(time.time() - start_time), 'sec')
    df['word_c_m_cc'] = (w1 * w2).sum(axis=1).astype('int8') # count of matches
    df['word_c_cs'] = np.nan_to_num(np.round((w1 * w2).sum(axis=1) / np.sqrt((w1 * w1).sum(axis=1) * (w2 * w2).sum(axis=1)), num_digits)).astype('float32') # cosine similarity



    # find words in name
    words_n = ['att', 'bank', 'burger', 'burgerking', 'cable', 'cafe', 'car', 'church', 'coffee', 'court', 'credit', 'cvs', 'depot', 'discount', 'dollar', 'domino', 'drug', 'dunkin', 'eleven', 'food', 'hardware', 'home', 'house', 'import', 'insurance', 'kfc', 'king', 'kitchen', 'mart', 'mcdonald', 'mobile', 'office', 'pizza', 'post', 'redbox', 'retail', 'sears', 'shack', 'shell', 'shoe', 'shop', 'sport', 'starbuck', 'station', 'stop', 'storage', 'store', 'subway', 'tax', 'travel', 'walmart']
    w1 = np.zeros([p1.shape[0], len(words_n)], dtype=np.int8)
    w2 = np.zeros([p2.shape[0], len(words_n)], dtype=np.int8)
    for i, word in enumerate(words_n): # currently 51
        w1[p1['name'].str.contains(word, regex=False), i] = 1
        w2[p2['name'].str.contains(word, regex=False), i] = 1
    print(i+1, 'finished words in name', int(time.time() - start_time), 'sec')
    df['word_n_m_cc'] = (w1 * w2).sum(axis=1).astype('int8') # count of matches
    df['word_n_cs'] = np.nan_to_num(np.round((w1 * w2).sum(axis=1) / np.sqrt((w1 * w1).sum(axis=1) * (w2 * w2).sum(axis=1)), num_digits)).astype('float32') # cosine similarity
    del w1, w2, ii
    gc.collect()



    # feature: count of ids within X meters of current. 140 seconds.
    lat  = np.array(train['latitude'])
    lon  = np.array(train['longitude'])
    lat1 = np.array((p1['latitude'] + p2['latitude'])/2)
    lon1 = np.array((p1['longitude'] + p2['longitude'])/2)

    @jit
    def count_close(lat, lon, lat1, lon1):
        cc = np.zeros([lat1.shape[0],7],dtype=np.int32)
        d0 = 2000**2 / 111111 ** 2
        d1 = 1000**2 / 111111 ** 2
        d2 = 500**2 / 111111 ** 2
        d3 = 200**2 / 111111 ** 2
        d4 = 100**2 / 111111 ** 2
        d5 = 50**2 / 111111 ** 2
        d6 = 5000**2 / 111111 ** 2
        for i in range(lat1.shape[0]):
            m = np.cos(lat1[i])**2
            dist2 = (lat - lat1[i]) ** 2 + m * (lon - lon1[i]) ** 2
            dist2 = dist2[dist2 < d6] # select subset of data to save time
            cc[i,0] = (dist2 < d0).sum()
            cc[i,1] = (dist2 < d1).sum()
            cc[i,2] = (dist2 < d2).sum()
            cc[i,3] = (dist2 < d3).sum()
            cc[i,4] = (dist2 < d4).sum()
            cc[i,5] = (dist2 < d5).sum()
            cc[i,6] = (dist2 < d6).sum()
        return cc

    cc = np.zeros([lat1.shape[0], 7], dtype=np.int32)
    step = 1
    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1) # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close(lat[idx2], lon[idx2], lat1[idx1], lon1[idx1])
            cc[idx1,:] = cc1
    cc = np.minimum(cc, 10000).astype(np.int16) # scale
    cc = (np.exp(np.round(np.log(cc+1), 1))-.5).astype(np.int16) # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df['id_cc_2K'] = cc[:,0]
    df['id_cc_1K'] = cc[:,1]
    df['id_cc_500'] = cc[:,2]
    df['id_cc_200'] = cc[:,3]
    df['id_cc_100'] = cc[:,4]
    df['id_cc_50'] = cc[:,5]
    df['id_cc_5K'] = cc[:,6]
    print('finished counting close points', int(time.time() - start_time), 'sec')



    # feature: count of ids of the same cat2 within X meters of current. 250 seconds.
    cat  = np.array(train['cat2'])
    cat1 = np.array(p1['cat2'])
    cat2 = np.array(p2['cat2'])

    @jit
    def count_close_cat(lat, lon, lat1, lon1, cat, cat1, cat2):
        cc = np.zeros([lat1.shape[0],7],dtype=np.int32)
        d0 = 2000**2 / 111111 ** 2
        d1 = 1000**2 / 111111 ** 2
        d2 = 500**2 / 111111 ** 2
        d3 = 200**2 / 111111 ** 2
        d4 = 100**2 / 111111 ** 2
        d5 = 50**2 / 111111 ** 2
        d6 = 5000**2 / 111111 ** 2
        for i in range(lat1.shape[0]):
            m = np.cos(lat1[i])**2
            dist2 = (lat - lat1[i]) ** 2 + m * (lon - lon1[i]) ** 2
            dist2 = dist2[(cat == cat1[i]) | (cat == cat2[i])] # only keep points for which cat matches
            dist2 = dist2[dist2 < d6] # select subset of data to save time
            cc[i,0] = (dist2 < d0).sum()
            cc[i,1] = (dist2 < d1).sum()
            cc[i,2] = (dist2 < d2).sum()
            cc[i,3] = (dist2 < d3).sum()
            cc[i,4] = (dist2 < d4).sum()
            cc[i,5] = (dist2 < d5).sum()
            cc[i,6] = (dist2 < d6).sum()
        return cc

    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1) # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close_cat(lat[idx2], lon[idx2], lat1[idx1], lon1[idx1], cat[idx2], cat1[idx1], cat2[idx1])
            cc[idx1,:] = cc1
    cc = np.minimum(cc, 10000).astype(np.int16) # scale
    cc = (np.exp(np.round(np.log(cc+1), 1))-.5).astype(np.int16) # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df['id_cc_cat_2K'] = cc[:,0]
    df['id_cc_cat_1K'] = cc[:,1]
    df['id_cc_cat_500'] = cc[:,2]
    df['id_cc_cat_200'] = cc[:,3]
    df['id_cc_cat_100'] = cc[:,4]
    df['id_cc_cat_50'] = cc[:,5]
    df['id_cc_cat_5K'] = cc[:,6]
    print('finished counting close cat points', int(time.time() - start_time), 'sec')



    # feature: count of ids of the same 'category_simpl' within X meters of current.
    cat  = np.array(train['category_simpl'])
    cat1 = np.array(p1['category_simpl'])
    cat2 = np.array(p2['category_simpl'])
    # if category_simpl = 0 then use categories
    train['cat_orig'] = train['categories'].astype('category').cat.codes + cat.max() + 10 # make sure this does not intersect with category_simpl codes
    cat[cat==0] = np.array(train['cat_orig'])[cat==0]
    p1 = p1.merge(train[['id','cat_orig']], on='id', how='left')
    p2 = p2.merge(train[['id','cat_orig']], on='id', how='left')
    cat1[cat1==0] = np.array(p1['cat_orig'])[cat1==0]
    cat2[cat2==0] = np.array(p2['cat_orig'])[cat2==0]
    train.drop('cat_orig', axis=1, inplace=True)

    for long_min in range(-180, 190, step):
        idx1 = (lon1 > long_min) & (lon1 < long_min + step)
        idx2 = (lon > long_min - 0.1) & (lon < long_min + step + 0.1) # margin
        if idx1.sum() > 0 and idx2.sum() > 0:
            cc1 = count_close_cat(lat[idx2], lon[idx2], lat1[idx1], lon1[idx1], cat[idx2], cat1[idx1], cat2[idx1])
            cc[idx1,:] = cc1
    cc = np.minimum(cc, 10000).astype(np.int16) # scale
    cc = (np.exp(np.round(np.log(cc+1), 1))-.5).astype(np.int16) # reduce cardinality - log-transform and round - 56 unique vals to 1000
    df['id_cc_simplcat_2K'] = cc[:,0]
    df['id_cc_simplcat_1K'] = cc[:,1]
    df['id_cc_simplcat_500'] = cc[:,2]
    df['id_cc_simplcat_200'] = cc[:,3]
    df['id_cc_simplcat_100'] = cc[:,4]
    df['id_cc_simplcat_50'] = cc[:,5]
    df['id_cc_simplcat_5K'] = cc[:,6]
    print('finished counting close simplcat points', int(time.time() - start_time), 'sec')


    # feature: compare numeric part of the name/address
    for col in ['name', 'address']:
        n1num = p1[col].apply(lambda x:"".join(re.findall("\d+", x)))
        n1num[n1num == ''] = '0'
        n1num = n1num.str[:9].astype('int32')
        n2num = p2[col].apply(lambda x:"".join(re.findall("\d+", x)))
        n2num[n2num == ''] = '0'
        n2num = n2num.str[:9].astype('int32')
        df[col+'_num'] = ((n1num != 0) * 1 + (n2num != 0) * 1 + ((n2num == n1num) & (n1num != 0)) * 2).astype('category') # 0/1/2/4
    del n1num, n2num
    gc.collect()


    # language combination of name
    df['langs'] = (np.minimum(p1['lang'], p2['lang']) + np.maximum(p1['lang'], p2['lang']) * 3).astype('category') # 6 possible combinations

    # cat_simpl, if matches; as cat.
    df['cat_simpl'] = p1['category_simpl'].astype('int16')
    df['cat_simpl'].iloc[df['same_cat_simpl'] == 0] = 0 # not a match - make it 0
    df['cat_simpl'] = df['cat_simpl'].astype('category')

    # drop skipped columns
    df.drop(list(f_skip0.intersection(df.columns)), axis=1, inplace=True)
    df.drop('same_cat_simpl', axis=1, inplace=True)
    gc.collect()
    return df
# FE2***************************************************************************************************************************************

df = FE2(df, p1, p2) # call FE2

#drop unneeded cols to save RAM - only keep id, reload the rest later
p1 = p1[['id']]
p2 = p2[['id']]

features = df.columns
types = df.dtypes
df = np.array(df, dtype=np.float32) # turn into np array to avoid memory spike


# print size of large vars
for i in dir():
    ss = sys.getsizeof(eval(i))
    if ss > 1000000:
        print(i, np.round(ss/1000000,1 ))

# lgb3 - the final one
print('data size is', df.shape)
cat_cols = []
for i, col in enumerate(features):
    if types[col]== 'category':
        cat_cols.append(i)
folds = 10 # number of folds
oof_preds = np.zeros(df.shape[0], dtype=np.float32)
#kf = KFold(n_splits=folds, shuffle=False)#, random_state=13)
#kf = GroupKFold(n_splits=folds)
kf = StratifiedKFold(n_splits=folds)
fi = np.zeros(df.shape[1])
#g = np.array(p1[['id']].merge(train[['id','point_of_interest']], on='id', how='left')['point_of_interest'])
for fold, (train_idx, valid_idx) in enumerate(kf.split(df, y)):
#for fold, (train_idx, valid_idx) in enumerate(kf.split(df, y+2*np.minimum(15,df[:,6])+2*100*np.minimum(11,df[:,3]))):
    x_train, x_valid = df[train_idx], df[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=cat_cols)
    lgb_valid = lgb.Dataset(x_valid, y_valid, categorical_feature=cat_cols)
    params = {'device':'gpu', 'seed':13, 'objective':'binary', 'verbose':-1, 'num_leaves':511, 'learning_rate':0.05, 'feature_fraction':.5
          , 'lambda_l1':1, 'lambda_l2':70, 'min_data_in_leaf':2000, 'min_gain_to_split':0.02, 'min_sum_hessian_in_leaf':0.03, 'path_smooth':0.2
          , 'min_data_in_bin':32}
    model = lgb.train(params, lgb_train, 20000, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=150, verbose_eval=20000)
    oof_preds[valid_idx] = model.predict(x_valid).ravel()
    fi += model.feature_importance(importance_type='gain', iteration=model.best_iteration)
    m = model.save_model('model_c'+str(fold)+'.txt') # save model
del x_train, x_valid, train_idx, valid_idx, y_train, y_valid
gc.collect()
# fi
fi = fi / fi.sum()
feature_imp = pd.DataFrame(zip(fi, features, types), columns=['Value','Feature', 'Type'])
feature_imp = feature_imp.sort_values(by='Value').reset_index(drop=True)
feature_imp['Value'] = np.round(feature_imp['Value'], 5)
feature_imp['nvalues'] = 0
# convert back to dataframe - slow(ish)
df = pd.DataFrame(df) 
df.columns = features
for col in types.index:
    df[col] = df[col].astype(types[col])
for i in range(feature_imp.shape[0]):
    feature_imp['nvalues'].iloc[i] = df[feature_imp['Feature'].iloc[i]].nunique()
print( feature_imp[:10] )  # worst
print( feature_imp[-10:] ) # best
get_CV(p1, p2, y, oof_preds) # get CV



# save predictions for post-processing
df2 = p1[['id']]
df2['id2'] = p2['id']
df2['y'] = y
df2['match'] = oof_preds.astype('float32')
df2.columns = ['id1', 'id2', 'y', 'oof_preds']
df2.to_csv('preds.csv', index=False)


# add name1                                                     GB=0.9213 0.55 CV*** 0.8883 9480 sec cv+=1.8 LB59=0.919
# take v61 Cat_solo_score/cat_link_score/cat_link_score_all(251)GB=0.9217 0.55 CV*** 0.8880 9887 sec cv-0.3 LB60=0.919
# sub with new lcs code:                                                                                    LB61=0.919  (ran 2 20)
# "The column "categories_split" must be created at the start, before the cleaning of the column "categories"; drop spaces
#                                                               GB=0.9225 0.55 CV*** 0.8893 9705 sec cv+1.3 LB62=0.920
# all: 3.4 Mil -> 3.8 Mil, 0.8894 -> 0.8886, 0.55 CV*** 0.8886 14861 sec (4.2 hours) cv-0.8

# GB base: 0.5 CV*** 0.9225 117 sec 0.5 CV*** 0.9231 161 sec
# Group names: replaces name with grouped name: 0.55 CV*** 0.9233 114 sec 0.5 CV*** 0.9232 158 sec - looks like a wash
# Group cities, states(253): 0.55 CV*** 0.9232 115 sec 0.55 CV*** 0.924 159 sec - looks promising. Size: 358050 9272 7925
# add more candidates(slower): Size: 381204 9309 7943 0.55 CV*** 0.9233 142 sec 0.55 CV*** 0.9243 187 sec - looks promising
# Add close candidates: Size:381204 9309 7943 (this code only applies to some countries)
# 0.55 CV*** 0.8904 12027 sec ...
# drop grouped state(252); use inference on additional p1/p2;        GB=0.55 CV*** 0.9241 147 sec/0.55 CV*** 0.9270 159 sec(leakage!)
#               0.55 CV*** 0.8901 11909 sec(3.3 h) cv+0.8

# Maybe you can try the new cat_link version, see if it changes something. GB=0.55 CV*** 0.924 139 sec
# use new file - code is the same. GB=0.5 CV*** 0.922 138 sec
#                                                               0.55 CV*** 0.8902 12459 sec - a wash. LB75=0.924 - 3rd best
# 80% of the data 0.55 CV*** 0.8974 8178 sec cv-2.8
# new link_between_categories/solo_cat_score                    GB=0.55 CV*** 0.9225 135 sec 0.55 CV*** 0.8902 11458 sec
#   models_39, LB=...
# skip 4 vars
# add features on 'name_initial','name_initial_decode'(295):    GB=0.55 CV*** 0.9237 143 sec 0.55 CV*** 0.8910 13291 sec cv+0.8 - good.
# drop 2(293), round ratiodist                                  GB=0.55 CV*** 0.9238 141 sec 0.55 CV*** 0.8911 12480 sec - good
# cat_initial(338)                                              GB=0.55 CV*** 0.9240 172 sec 0.55 CV*** 0.8910 14558 sec - worse, undo****
# remove spaces from initial  (l 1939)                          GB=0.55 CV*** 0.9235 134 sec 0.55 CV*** 0.8910 12205 sec - worse, undo
# round more floats (cat_link_score_all)                        GB=0.55 CV*** 0.9236 145 sec 0.55 CV*** 0.8913 12312 sec LB83=0.927
# catboost                                                      GB=0.55 CV*** 0.9191(5) 609 sec 0.9184(4) did not finish - oom!
# round more vars                                               GB=0.55 CV*** 0.9240 135 sec 0.55 CV*** 0.8912 12395 sec - a wash, accept
# drop 20 vars (271)                                            GB=0.55 CV*** 0.9244 133 sec 0.55 CV*** 0.8911 12202 sec - a wash
# drop more vars, drop high corr vars5,(243)                    GB=0.55 CV*** 0.9241 140 sec 0.55 CV*** 0.8912 13326 sec - a wash, accept LB84=0.927
# drop 30(213)                                                  GB=0.55 CV*** 0.9240 138 sec 0.55 CV*** 0.8908 11858 sec - worse, undo
# russian translate RU=0.55 CV*** 0.8624 622 sec 0.55 CV*** 0.863 608 sec - looks slightly better, accept
# 5 folds on model 1                                            GB=0.55 CV*** 0.9249 177 sec 0.55 CV*** 0.8912 12474 sec LB85=0.927 - second best
# decrease early stopping(30) (hardcode iter limit lower?)                                   0.55 CV*** 0.8912 12168 sec LB86=0.927 - third best - undo
# new params(high l2)                                           GB=0.55 CV*** 0.9219 150 sec 0.55 CV*** 0.8915 15169 sec LB89=0.928 - best LB *********************
# 10 folds                                                      GB=0.55 CV*** 0.9243 185 sec 0.55 CV*** 0.8931 24140 sec cv+1.6 LB91=...
# not-stratified kfold in lgb3                                  GB=0.55 CV*** 0.9269 209 sec 0.55 CV*** 0.8975 27993 sec cv+4.4 LB97=0.933
# Kfold 10 folds trim_mean(0.1), fast inference(1%+fold*1%)
# all Stratified, y only                                        GB=0.55 CV*** 0.9231 180 sec 0.55 CV*** 0.8931 23971 sec - ???
# shufle, stratified(y)                                         GB=0.50 CV*** 0.9256 194 sec 0.55 CV*** 0.8981 30045 sec LB99=...
# shufle, all stratified                                        ... 



# add model 2
# reduce cut1 from .007 to .004
# tune params
# more folds
# try median, or mean of meadian 50% of results
