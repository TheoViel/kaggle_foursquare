import gc
import random
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from params import DEBUG, OUT_PATH, IS_TEST
from ressources import COUNTRIES
from matching import (
    load_cleaned_data,
    print_infos,
    lcs,
    lcs2,
    distance,
    pi1,
    substring_ratio,
    subseq_ratio,
    ll_lcs,
    get_CV,
    Compute_Mdist_Mindex,
    vectorisation_similarite,
    haversine,
    create_address,
    find_potential_matchs,
)

random.seed(13)
warnings.simplefilter("ignore")


# ## Load Data

# In[5]:


if IS_TEST:
    train = load_cleaned_data(OUT_PATH + "cleaned_data_test.csv")
else:
    train = load_cleaned_data(OUT_PATH + "cleaned_data_train.csv")


# In[6]:


if DEBUG:
    train = train.head(10000)


# ### Target

# In[7]:


if not IS_TEST:
    clusts = (
        train[["id", "point_of_interest"]]
        .groupby("point_of_interest")
        .agg(list)
        .reset_index()
    )
    clusts = clusts[clusts["id"].apply(lambda x: len(x) > 1)]

    N_TO_FIND = clusts["id"].apply(lambda x: len(x)).sum()
    print(N_TO_FIND)

    example = clusts.explode("id").sort_values("id")
    example["y"] = 1

    print_infos(example, None, N_TO_FIND)

    p1 = example.sample(len(example) // 2)

    print_infos(p1, None, N_TO_FIND)

else:
    N_TO_FIND = -1


# ## Matching

# In[8]:


p3 = train[["country", "id", "point_of_interest", "phone", "lon2"]].copy()
p3 = (
    p3.loc[p3["phone"] != ""]
    .sort_values(by=["country", "phone", "lon2", "id"])
    .reset_index(drop=True)
)


# ### Phone

# In[9]:


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


# In[10]:


print_infos(p1, p2, N_TO_FIND)


# ### Lat / lon 22m² square

# In[11]:


# lat/lon, rounded to 2 x 4 digits = 22* meters square;
# there should not be too many false positives this close to each other
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


# In[12]:


print_infos(p1, p2, N_TO_FIND)


# ### Url

# In[13]:


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


# In[14]:


print_infos(p1, p2, N_TO_FIND)


# ### Categories

# In[15]:


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


# In[16]:


print_infos(p1, p2, N_TO_FIND)


# ### Address

# In[17]:


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


# In[18]:


print_infos(p1, p2, N_TO_FIND)


# ### Name

# In[19]:


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


# In[20]:


print_infos(p1, p2, N_TO_FIND)


# ### Latitude

# In[21]:


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

p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)


# In[22]:


print_infos(p1, p2, N_TO_FIND)


# ### Longitude

# In[23]:


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


# In[24]:


print_infos(p1, p2, N_TO_FIND)


# In[25]:


p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "Added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
)


# ## Name lcs
# - Slow

# In[26]:


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


# In[27]:


idx1 = []
idx2 = []

for i in tqdm(range(p3.shape[0] - 1)):
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

p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)


# In[28]:


print_infos(p1, p2, N_TO_FIND)


# In[29]:


p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "Added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
)


# ### Clean

# In[30]:


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


# In[31]:


print_infos(p1, p2, N_TO_FIND)


# In[32]:


p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)
a = p1.groupby("id")["y"].sum().reset_index()
print(
    "Added",
    p1a.shape[0],
    p1["y"].sum(),
    np.minimum(1, a["y"]).sum(),
)


# In[33]:


del d, names, lon2, p3, idx1, idx2, p1a, p2a, lat, lon
gc.collect()


# ## Extend

# ### Sort to put similar points next to each other - for constructing pairs

# In[34]:


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


# ## Construct pairs

# In[35]:


cols = [
    "id",
    "latitude",
    "longitude",
    "point_of_interest",
    "name",
    "category_simpl",
    "name_initial_decode",
]
colsa = ["id", "point_of_interest"]

p1a = train[colsa].copy()
p2a = train[colsa].iloc[1:, :].reset_index(drop=True).copy()
p2a = p2a.append(train[colsa].iloc[0], ignore_index=True)

p1 = p1.append(p1a, ignore_index=True)
p2 = p2.append(p2a, ignore_index=True)


# In[36]:


print_infos(p1, p2, N_TO_FIND)


# In[37]:


p1_svg = p1.copy()
p2_svg = p2.copy()


# ### Add more shifts
# - Slow (15min)

# In[38]:


p1 = p1_svg.copy()
p2 = p2_svg.copy()


# In[ ]:


for i, s in enumerate(tqdm(range(2, 121))):  # 121
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

    if not (s % 10):
        # get stats; overstated b/c dups are not excluded yet
        print(f"{i}, s={s}")
        print_infos(p1, p2, N_TO_FIND)
        print()

    gc.collect()


# In[ ]:


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


# In[ ]:


print_infos(p1, p2, N_TO_FIND)


# In[ ]:


get_CV(
    p1,
    p2,
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    train,
)


# ### Add close candidates
# - Slow

# In[ ]:


already_found = {
    tuple(sorted([idx1, idx2])): 1 for idx1, idx2 in zip(p1["id"], p2["id"])
}

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
                if key not in already_found:
                    key_idx = tuple(sorted([Original_idx[idx1], Original_idx[idx2]]))
                    try:
                        if key_idx not in new_cand:
                            new_true_match += int(infos[idx1, -1] == infos[idx2, -1])
                    except Exception:
                        pass
                    new_cand.add(key_idx)

    # Add new candidates
    New_candidates += [list(x) for x in new_cand]
    print(
        f"Country {country_} ({COUNTRIES[country_-1]}) : {new_true_match}/{len(new_cand)} added."
    )


# In[ ]:


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

print(f"Candidates added : {len(p1) - size1}/{len(p1)}.")


# In[ ]:


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


# In[49]:


print_infos(p1, p2, N_TO_FIND)


# In[50]:


get_CV(
    p1,
    p2,
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    train,
)


# In[ ]:


p1_svg = p1.copy()
p2_svg = p2.copy()


# ### Add close candidates v2
# - TODO

# In[51]:


p1 = p1_svg.copy()
p2 = p2_svg.copy()

train["lat2"] = np.round(train["latitude"], 0).astype("int8")
train["lon2"] = np.round(train["longitude"], 0).astype("int8")

# sort to put similar points next to each other - for constructing pairs
sort = [
    "category_simpl",
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

maxdist = (train["q90"] * 400 + train["q99"] * 400).to_numpy()


# In[52]:

# add more shifts, only for short distances or for partial name matches
for i, s in enumerate(tqdm(range(1, 50))):  # 121

    s2 = s  # shift
    p2a = train[cols].iloc[s2:, :]
    p2a = p2a.append(train[cols].iloc[:s2, :], ignore_index=True)

    # drop pairs with large distances
    same_cat_simpl = (train['category_simpl'] == p2a['category_simpl']).to_numpy()

    dist = distance(
        np.array(train['latitude']),
        np.array(train['longitude']),
        np.array(p2a['latitude']),
        np.array(p2a['longitude'])
    )

    ii = np.zeros(train.shape[0], dtype=np.int8)
    x1 = train[['name', 'name_initial_decode']].to_numpy()
    x2 = p2a[['name', 'name_initial_decode']].to_numpy()

    for j in range(train.shape[0]):  # pi1 adds 14K matches

        if same_cat_simpl[j] and dist[j] < maxdist[j]:

            name1, name2 = x1[j][0], x2[j][0]
            name_ini1, name_ini2 = x1[j][1], x2[j][1]

            if pi1(name1, name2) == 1:
                ii[j] = 1
            elif substring_ratio(name1, name2) >= 0.6:
                ii[j] = 1
            elif subseq_ratio(name1, name2) >= 0.7:
                ii[j] = 1
            elif len(name1) >= 6 and len(name2) >= 6 and name1.endswith(name2[-6:]):
                ii[j] = 1

            # elif has_common_word(name_ini1, name_ini2, min_len=6) :
            #    ii[j] = 1
            # elif word_in_common(name_ini1, name_ini2, min_len_word=6):
            #    ii[j] = 1
            # elif subword_in_common(name_ini1, name_ini2, min_len_word=6) :
            #    ii[j]=1

    idx = (ii > 0)

    p1 = p1.append(train[colsa].loc[idx], ignore_index=True)
    p2 = p2.append(p2a[colsa].loc[idx], ignore_index=True)

    if s < 10 or s % 10 == 0:
        # get stats; overstated b/c dups are not excluded yet
        print(f"{i}, s={s}")
        print_infos(p1, p2, N_TO_FIND)
        print()

del p2a, dist, idx
gc.collect()

# In[53]:


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


# In[54]:


print_infos(p1, p2, N_TO_FIND)


# In[55]:


get_CV(
    p1,
    p2,
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    train,
)


# In[56]:


p1_svg = p1.copy()
p2_svg = p2.copy()


# ### Candidates in initial Youri's solution

# In[57]:


p1["y"] = np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8)

ID_to_POI = dict(zip(train["id"], train["point_of_interest"]))
nb_true_matchs_initial = 0
Cand = {}
for i, (id1, id2) in enumerate(zip(p1["id"], p2["id"])):
    key = f"{min(id1, id2)}-{max(id1, id2)}"
    Cand[key] = p1["y"].iloc[i]
    nb_true_matchs_initial += int(ID_to_POI[id1] == ID_to_POI[id2])


# ### TF-IDF n°1 : airports

# In[58]:


p1 = p1_svg.copy()
p2 = p2_svg.copy()


# In[59]:


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
                        Added_p1.append([id1, poi1])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added for tfidf n°1 (airports) : {len(p1)-size1}/{len(p1)}.")


# In[60]:


print_infos(p1, p2, N_TO_FIND)


# ### TF-IDF n°2 : metro stations

# In[61]:


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
                        Added_p1.append([id1, poi1])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        print(
            f"Candidates added for tfidf n°2 (metro stations) : {len(p1)-size1}/{len(p1)}."
        )


# In[62]:


print_infos(p1, p2, N_TO_FIND)


# ### TF-IDF n°3a : for each countries (with initial unprocessed name)

# In[63]:


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
    print(f"# Country n°{country} : {COUNTRIES[country-1]}.")

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
                        Added_p1.append([id1, poi1])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added : {len(p1)-size1}/{len(p1)}.")

print("\n-> TF-IDF for contries finished.")
print(f"Candidates added : {len(p1)-size}.")


# In[64]:


print_infos(p1, p2, N_TO_FIND)


# ### TF-IDF n°3b : for each countries (with few processed name)

# In[65]:


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
    print(f"# Country n°{country} : {COUNTRIES[country-1]}.")

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
                        Added_p1.append([id1, poi1])
                        Added_p2.append([id2, poi2])
                        nb_true_matchs_initial += int(poi1 == poi2)

        p1 = (
            p1.append(pd.DataFrame(Added_p1, columns=p1.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        p2 = (
            p2.append(pd.DataFrame(Added_p2, columns=p2.columns[:2]))
            .reset_index(drop=True)
            .copy()
        )
        print(f"Candidates added : {len(p1)-size1}.")

print("\n-> TF-IDF for contries finished.")
print(f"Candidates added : {len(p1)-size}.")


# In[66]:


print_infos(p1, p2, N_TO_FIND)


# ### Add candidates based on same name/phone/address

# In[67]:


# Création d'un df de travail
work = train.copy()

# Prepare columns
for c in ["name", "address", "city"]:
    work[c] = work[c].astype(str).str.lower()
work["index"] = work.index


work_names = work.groupby("name")["index"].apply(list).to_frame().reset_index()
work_names = dict(zip(work_names["name"], work_names["index"]))
work_names = {
    name: Liste_idx
    for name, Liste_idx in work_names.items()
    if len(name) >= 2 and len(Liste_idx) <= 25
}  # Don't consider too widespread names


work_phones = work.groupby("phone")["index"].apply(list).to_frame().reset_index()
work_phones = dict(zip(work_phones["phone"], work_phones["index"]))
work_phones = {
    phone: Liste_idx
    for phone, Liste_idx in work_phones.items()
    if len(phone) >= 3 and len(Liste_idx) <= 10
}  # Don't consider too widespread phone


work["address_complet"] = work.apply(
    lambda row: create_address(row["address"], row["city"]), axis=1
)
work_address = (
    work.groupby("address_complet")["index"].apply(list).to_frame().reset_index()
)
work_address = dict(zip(work_address["address_complet"], work_address["index"]))
work_address = {
    address: Liste_idx
    for address, Liste_idx in work_address.items()
    if len(address) >= 3 and len(Liste_idx) <= 10
}  # Don't consider too widespread address


# In[68]:


# Process
# tqdm.pandas()
# Potential_on_NamePhone = work.progress_apply(find_potential_matchs, axis=1).to_list()
Potential_on_NamePhone = work.apply(
    lambda row: find_potential_matchs(row, work_names, work_phones, work_address),  # noqa
    axis=1,
).to_list()

# Don't keep pairs too far from each other
Potential_on_NamePhone_new = []

# Numpy for faster process
train_numpy = train[["name", "latitude", "longitude", "category_simpl"]].to_numpy()

# Filtre on dist
for i, Liste_idx in enumerate(Potential_on_NamePhone):
    new = []
    name1, lat1, lon1, cat_simpl1 = (
        train_numpy[i][0],
        train_numpy[i][1],
        train_numpy[i][2],
        train_numpy[i][3],
    )
    for j, row in enumerate(train_numpy[Liste_idx]):
        name2, lat2, lon2, cat_simpl2 = row[0], row[1], row[2], row[3]

        # if rare name, we are more tolerant
        if name1 == name2 and len(work_names[name1]) <= 5:
            thr_distance = 100
        else:
            thr_distance = 26

        # if the category is usually far even for matchs
        if (cat_simpl1 in far_cat_simpl) or (cat_simpl2 in far_cat_simpl):
            thr_distance = 350
            if (cat_simpl1 == 1) or (cat_simpl2 == 1):
                thr_distance = 100000  # no limit

        # Add distance if long names (not a coincidence if they are equal)
        if name1 == name2 and len(name1) >= 10:
            thr_distance += 15

        # Process
        if haversine(lat1, lon1, lat2, lon2) > thr_distance:
            continue
        else:
            new.append(Liste_idx[j])
    Potential_on_NamePhone_new.append(new.copy())

Potential_on_NamePhone = Potential_on_NamePhone_new.copy()

del Potential_on_NamePhone_new, train_numpy
gc.collect()

# Number of potential matchs
print(
    f"Potential match on name/phone/address : {sum(len(x) for x in Potential_on_NamePhone)}."
)


# In[69]:


# Add matches
size1 = len(p1)
Added_p1, Added_p2 = [], []


seen = set()
for idx1, Liste_idx in enumerate(Potential_on_NamePhone):

    id1, lat1, lon1, cat1, cat_simpl1 = (
        train["id"].iat[idx1],
        train["latitude"].iat[idx1],
        train["longitude"].iat[idx1],
        train["categories"].iat[idx1],
        train["category_simpl"].iat[idx1],
    )
    for idx2 in Liste_idx:
        if idx1 != idx2:
            id2, lat2, lon2, cat2, cat_simpl2 = (
                train["id"].iat[idx2],
                train["latitude"].iat[idx2],
                train["longitude"].iat[idx2],
                train["categories"].iat[idx2],
                train["category_simpl"].iat[idx2],
            )
            key = f"{min(id1, id2)}-{max(id1, id2)}"
            # same_cat = (cat_simpl1==cat_simpl2 and cat_simpl1>0) or (cat1==cat2 and cat1!='')
            if key not in Cand and key not in seen:
                seen.add(key)
                poi1, poi2 = (
                    train["point_of_interest"].iat[idx1],
                    train["point_of_interest"].iat[idx2],
                )
                Added_p1.append([id1, poi1])
                Added_p2.append([id2, poi2])

p1 = (
    p1.append(pd.DataFrame(Added_p1, columns=p1.columns[:2]))
    .reset_index(drop=True)
    .copy()
)
p2 = (
    p2.append(pd.DataFrame(Added_p2, columns=p2.columns[:2]))
    .reset_index(drop=True)
    .copy()
)
print(f"Candidates added with name/phone/address similarity : {len(p1)-size1}.")

del (
    Tfidf_idx,
    Tfidf_val,
    Cand,
    ID_to_POI,
    Potential_on_NamePhone,
    work,
    work_names,
    work_phones,
    work_address,
    Added_p1,
    Added_p2,
    seen,
    Names_numrow,
)
gc.collect()


# In[70]:


print_infos(p1, p2, N_TO_FIND)


# ### Final PP

# In[71]:


# Reset index after Vincent's candidate addition
p1 = p1.reset_index(drop=True)
p2 = p2.reset_index(drop=True)

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


# In[72]:


print_infos(p1, p2, N_TO_FIND)


# In[73]:


get_CV(
    p1,
    p2,
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    np.array(p1["point_of_interest"] == p2["point_of_interest"]).astype(np.int8),
    train,
)


# In[74]:


if not DEBUG:
    if IS_TEST:
        p1.to_csv(OUT_PATH + "p1_yv_test.csv", index=False)
        p2.to_csv(OUT_PATH + "p2_yv_test.csv", index=False)
    else:
        p1.to_csv(OUT_PATH + "p1_yv_train.csv", index=False)
        p2.to_csv(OUT_PATH + "p2_yv_train.csv", index=False)

print("Done !")
