import gc
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from unidecode import unidecode

from params import DATA_PATH, OUT_PATH, RESSOURCES_PATH, IS_TEST, DEBUG, N_ROWS
from ressources import (
    NAME_DI,
    CITY_DI,
    CITY_DI_2,
    LL1,
    ALL_WORDS,
    CAT2_DI,
    CAT_REGROUP,
    COUNTRIES,
    ID_DI,
    L1S_ID
)

from cleaning import (
    isEnglish,
    convert_japanese_alphabet,
    find_cat,
    replace_seven_eleven,
    replace_seaworld,
    replace_mcdonald,
    simplify_cat,
    st,
    apply_solo_cat_score,
    apply_cat_distscore,
    rem_expr,
    rem_abr,
    get_caps_leading,
    rem_words,
    clean_nums,
    clean_address,
    st2,
    process_phone,
    id_translate,
    replace_common_words,
)

random.seed(13)
warnings.simplefilter("ignore")


# ## Load Data

# In[17]:


if IS_TEST:
    train = pd.read_csv(DATA_PATH + "test.csv")
    train["point_of_interest"] = 0
else:
    train = pd.read_csv(DATA_PATH + "train.csv")


if DEBUG:
    train = train.head(N_ROWS)

# ## Cleaning & processing

# ### Language

# In[18]:


train["lang"] = train["name"].apply(isEnglish).astype("int8")


# In[19]:


idx_JP = train[train["country"] == "JP"].index
train.loc[idx_JP] = convert_japanese_alphabet(train.loc[idx_JP])


# ### Fill-in missing categories, based on words in name

# In[20]:


Key_words_for_cat = pd.read_pickle(RESSOURCES_PATH + "dict_for_missing_cat.pkl")

train["categories"] = train["categories"].fillna("")
idx_missing_cat = train[train["categories"] == ""].index
train.loc[idx_missing_cat, "categories"] = (
    train.loc[idx_missing_cat, "name"]
    .fillna("")
    .apply(lambda x: find_cat(x, Key_words_for_cat))
)
del idx_missing_cat
gc.collect()


# ### Pre-format data

# In[21]:


train["point_of_interest"] = (
    train["point_of_interest"].astype("category").cat.codes
)  # turn POI into ints to save spacetime
train["latitude"] = train["latitude"].astype("float32")
train["longitude"] = train["longitude"].astype("float32")


# ### Sorted by count in candidate training data

# In[22]:


c_di = {}
for i, c in enumerate(COUNTRIES):  # map train/test countries the same way
    c_di[c] = min(
        50, i + 1
    )  # cap country at 50 - after that there are too few cases per country to split them
train["country"] = (
    train["country"].fillna("ZZ").map(c_di).fillna(50).astype("int16")
)  # new country maps to missing (ZZ)


# ### GT

# In[23]:


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


# ### Copy

# In[24]:


train["name_svg"] = train["name"].copy()
train["categories_svg"] = train["categories"].copy()


# ### Clean name

# In[25]:


train["name"] = train["name"].apply(lambda x: unidecode(str(x).lower()))
train["name"] = train["name"].apply(lambda text: replace_seven_eleven(text))
train["name"] = train["name"].apply(lambda text: replace_seaworld(text))
train["name"] = train["name"].apply(lambda text: replace_mcdonald(text))


# ### Simple category

# In[26]:


train["category_simpl"] = (
    train["categories"]
    .astype(str)
    .apply(lambda text: simplify_cat(text, CAT_REGROUP))
    .astype("int16")
)

print(
    "Simpl categories found :", len(train[train["category_simpl"] > 0]), "/", len(train)
)


# ### Go back to initial columns

# In[27]:


train["name"] = train["name_svg"].copy()
train["categories"] = train["categories_svg"].copy()
train.drop(["name_svg", "categories_svg"], axis=1, inplace=True)


# ### Save names separated by spaces for tf-idf

# In[28]:


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


# ### Find the score of the categories

# In[29]:


cat_pairings = pd.read_pickle(
    RESSOURCES_PATH + "howmanytimes_groupedcat_are_paired_with_other_groupedcat.pkl"
)  # link-between-grouped-cats

# Find the score of the categories
train["freq_pairing_with_other_groupedcat"] = (
    train["category_simpl"].apply(lambda cat: cat_pairings[cat]).fillna(0)
)


# In[30]:


solo_cat_scores = pd.read_pickle(
    RESSOURCES_PATH + "solo_cat_score.pkl"
)  # link-between-categories - 1858 values

train["cat_solo_score"] = (
    train["categories_split"]
    .apply(lambda List_cat: apply_solo_cat_score(List_cat, solo_cat_scores))
    .fillna(0)
)


# In[31]:


# Find the scores
Dist_quantiles = pd.read_pickle(
    RESSOURCES_PATH + "Dist_quantiles_per_cat.pkl"
)  # dist-quantiles-per-cat - 869 values

col_cat_distscores = ["Nb_multiPoi", "mean", "q25", "q50", "q75", "q90", "q99"]
train.loc[:, col_cat_distscores] = (
    train["categories_split"]
    .apply(lambda x: apply_cat_distscore(x, Dist_quantiles))
    .to_list()
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


# ### Clean name

# In[32]:


# remove some expressions from name
train["name"] = train["name"].apply(rem_expr)

# drop abbreviations all caps in brakets for long enough names
train["name"] = train["name"].apply(rem_abr)

# select capitals only, or first letter of each word (which could have been capital)
train["nameC"] = train["name"].fillna("").apply(get_caps_leading)


# ### More cleaning
# - A bit slow

# In[33]:


for col in tqdm(["name", "address", "city", "state", "zip", "url", "categories"]):
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


# ### Clean Name again

# In[34]:


# fix some misspellings
for w in NAME_DI.keys():
    train["name"] = train["name"].apply(lambda x: x.replace(w, NAME_DI[w]))

# new code from V *************************************************************
# Group names
name_groups = pd.read_pickle(RESSOURCES_PATH + "name_groups.pkl")
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
del name_groups
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


# ### Replace some cities

# In[35]:


for key in CITY_DI.keys():
    train["city"].loc[train["city"] == key] = CITY_DI[key]
# second pass

for key in CITY_DI_2.keys():
    train["city"].loc[train["city"] == key] = CITY_DI_2[key]


# ### Group cities & state

# In[36]:


# City group
with open(RESSOURCES_PATH + "dict_for_city_groups.pkl", "rb") as f:
    dict_for_city_groups = pickle.load(f)

train["city_group"] = train["city"].apply(
    lambda city: dict_for_city_groups.get(city, city)
)

# State group
with open(RESSOURCES_PATH + "dict_for_state_groups.pkl", "rb") as f:
    dict_for_state_groups = pickle.load(f)

train["state_group"] = train["state"].apply(
    lambda state: dict_for_state_groups.get(state, state)
)


# ### Clean city

# In[37]:


# cap length at 38
train["city"] = train["city"].str[:38]

# eliminate some common words that do not change meaning
for w in ["gorod"]:
    train["city"] = train["city"].apply(lambda x: x.replace(w, ""))

train["city"].loc[train["city"] == "nan"] = ""


# ### Clean address

# In[38]:


train["address"].loc[train["address"] == "nan"] = ""
train["address"] = train["address"].str[:99]  # cap length at 99
train["address"] = train["address"].apply(lambda x: x.replace("street", "str"))


# ### Clean state

# In[39]:


train["state"] = train["state"].str[:33]  # cap length at 33
state_di = {
    "calif": "ca",
    "jakartacapitalregion": "jakarta",
    "moscow": "moskva",
    "seoulteugbyeolsi": "seoul",
}
for key in state_di.keys():
    train["state"].loc[train["state"] == key] = state_di[key]
train["state"].loc[train["state"] == "nan"] = ""


# ### Clean url

# In[40]:


train["url"] = train["url"].str[:129]  # cap length at 129
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


# ### Clean phone

# In[41]:


train["phone"] = train["phone"].apply(lambda text: process_phone(text))
# all matches of last 9 digits look legit - drop leading digit
train["phone"] = train["phone"].str[1:]
# set invalid numbers to empty
idx = (train["phone"] == "000000000") | (train["phone"] == "999999999")
train["phone"].loc[idx] = ""


# ### Clean categories

# In[42]:


train["categories"] = train["categories"].str[:68]  # cap length at 68
train["categories"].loc[train["categories"] == "nan"] = ""
cat_di = {"aiport": "airport", "terminal": "airport"}
for key in cat_di.keys():
    train["categories"] = train["categories"].apply(
        lambda x: x.replace(key, cat_di[key])
    )


# ### Translation

# In[43]:


# Translate Indonesian
idx = train["country"] == 2  # ID

for col in ["name", "address", "city", "state"]:
    train[col].loc[idx] = train[col].loc[idx].apply(lambda x: id_translate(x, ID_DI))


# In[44]:


# # translate russian words
# dict_ru_en = pd.read_pickle(RESSOURCES_PATH + "dict_translate_russian.pkl")

# idx = train["country"] == 6  # RU
# for k in ["city", "state", "address", "name"]:
#     train.loc[idx, k] = (
#         train.loc[idx, k]
#         .astype(str)
#         .apply(lambda x: translate_russian_word_by_word(x, dict_ru_en))
#     )
#     train.loc[idx, k] = train.loc[idx, k].apply(lambda x: "" if x == "nan" else x)
# del dict_ru_en


# ### Replacing words

# In[45]:


# match some identical names - based on analysis of mismatched names for true pairs
# soekarno-hatta international airport - Jakarta, ID

idx = train["country"] == 2  # ID - this is where this location is

for l1 in L1S_ID:
    train["name"].loc[idx] = (
        train["name"].loc[idx].apply(lambda x: x if x not in l1[1:] else l1[0])
    )


# In[46]:


for l1 in LL1:
    train["name"] = train["name"].apply(lambda x: x if x not in l1[1:] else l1[0])


# In[47]:


train["name"] = train["name"].apply(replace_common_words)


# ### Cat 2

# In[48]:


# define cat2 (clean category with low cardinality)
# base it on address, name and catogories - after those have been cleaned
# (then do not need to include misspellings)
train["cat2"] = ""  # init (left: 129824*)

for col in ["address", "categories", "name"]:
    for word in ALL_WORDS.keys():
        words = ALL_WORDS[word]
        for w in words:
            train["cat2"].loc[train[col].str.contains(w, regex=False)] = word

train["cat2"] = train["cat2"].map(CAT2_DI).astype("int16")


# ### Nans

# In[49]:


for c in [
    "id",
    "name",
    "address",
    "city",
    "state",
    "zip",
    "url",
    "phone",
    "categories",
    "m_true",
    "categories_split",
    "name_initial",
    "name_initial_decode",
    "nameC",
    "name2",
    "city_group",
    "state_group",
]:
    train.loc[train[c] == "null", c] = ""
    train.loc[train[c] == "nan", c] = ""


# ## Save

# In[50]:


if IS_TEST:
    train.to_csv(OUT_PATH + "cleaned_data_test.csv", index=False)
else:
    train.to_csv(OUT_PATH + "cleaned_data_train.csv", index=False)

print("Done !")
