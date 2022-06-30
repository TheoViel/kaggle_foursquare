import re
import pykakasi
import numpy as np
from unidecode import unidecode


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


def convert_japanese_alphabet(df):
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')  # Convert Hiragana into alphabet
    kakasi.setMode('K', 'a')  # Convert Katakana into alphabet
    kakasi.setMode('J', 'a')  # Convert Kanji into alphabet
    conversion = kakasi.getConverter()

    def convert(row):
        for column in ["name", "address", "city", "state"]:
            try:
                row[column] = conversion.do(row[column])
            except Exception:
                pass
        return row

    df = df.apply(convert, axis=1)
    return df


def process(cat, split=" "):
    cat = [x for x in str(cat).split(split) if cat != "" and len(x) >= 2]
    # Keep only letters
    cat = [re.sub(r"[^a-zA-Z]", " ", x) for x in cat]
    # Delete multi space
    cat = [re.sub("\\s+", " ", x).strip() for x in cat]
    return cat


# Function to fill missing categories
def find_cat(name, Key_words_for_cat):
    name_list = process(unidecode(str(name).lower()))
    for cat, wordlist in Key_words_for_cat.items():
        if any(name_word in name_list for name_word in wordlist):
            return cat
    return ""


def replace_seven_eleven(text):
    new = "seven eleven"
    for sub in ["7/11", "7-11", "7-eleven"]:
        text = text.replace(sub + "#", new + " ")
        text = text.replace(sub + " ", new + " ")
        text = text.replace(sub, new)
    return text


def replace_seaworld(text):
    new = "seaworld"
    for sub in ["sea world"]:
        text = text.replace(sub, new)
    return text


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


def process_text_cat(text):
    text = unidecode(text.lower())
    res = " ".join([re.sub(r"[^a-zA-Z]", " ", x).strip() for x in text.split()])
    return re.sub("\\s+", " ", res).strip()


def simplify_cat(categories, Cat_regroup):
    categories = str(categories).lower()
    if categories in ("", "nan"):
        return -1
    for cat in categories.split(","):
        cat = process_text_cat(cat)
        for i, Liste in enumerate(Cat_regroup):
            if any(cat == x for x in Liste):
                return i + 1
    else:
        return 0


def st(x, remove_space=False):
    # turn to latin alphabet
    x = unidecode(str(x))
    # lower case
    x = x.lower()
    # remove symbols
    x = x.replace('"', "")
    ss = ",:;'/-+&()!#$%*.|\@`~^<>?[]{}_=\n"  # noqa
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


def replace_common_words(s):
    for x in ["kebap", "kebab", "kepab", "kepap"]:
        s = s.replace(x, "kebab")
    for x in ["aloonaloon"]:  # center place in indonesian villages
        s = s.replace(x, "alunalun")
    for x in ["restoram"]:
        s = s.replace(x, "restaurant")
    s = s.replace("internationalairport", "airport")
    return s


def apply_solo_cat_score(List_cat, solo_cat_scores):
    return max([solo_cat_scores[cat] for cat in List_cat])


def apply_cat_distscore(List_cat, Dist_quantiles):
    q = np.array([Dist_quantiles[cat] for cat in List_cat if cat in Dist_quantiles])
    if len(q) == 0:
        return Dist_quantiles[""]
    return np.max(q, axis=0)


def rem_expr(x):
    x = str(x)
    x = x.replace("™", "")  # tm
    x = x.replace("®", "")  # r
    x = x.replace("ⓘ", "")  # i
    x = x.replace("©", "")  # c
    return x


def rem_abr(x):
    x = str(x)
    if "(" in x and ")" in x:  # there are brakets
        i = x.find("(")
        j = x.find(")")
        if j > i + 1 and j - i < 10 and len(x) - (j - i) > 9:  # remainder is long enough
            s = x[i + 1: j]
            # clean it
            ss = " ,:;'/-+&()!#$%*.|`~^<>?[]{}_=\n"
            for k in range(len(ss)):
                s = s.replace(ss[k], "")
            if s == s.upper():  # all caps (and/or numbers)
                x = x[:i] + x[j + 1:]
    return x


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
            x = x[len(word) + 1:]
        if x[-len(word) - 1:] == " " + word:  # end
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
            if x[-len(word) - 1:] == " " + word:  # end
                x = x.replace(" " + word, " " + words[0])
    return x


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


def id_translate(x, id_di):  # translate, and move some new words to the beginning
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


def process_text(x):
    return re.findall(r"\w+", x.lower().strip())


def translate_russian_word_by_word(text, dict_ru_en):
    text = process_text(text)
    text = [dict_ru_en[word] if word in dict_ru_en else word for word in text]
    return " ".join(text)
