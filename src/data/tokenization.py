from transformers import AutoTokenizer


def get_tokenizer(name, folder=None):
    if folder is None:
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(folder)

    tokenizer.name = name
    tokenizer.special_tokens = {
        "sep": tokenizer.sep_token_id,
        "cls": tokenizer.cls_token_id,
        "pad": tokenizer.pad_token_id,
    }

    return tokenizer
