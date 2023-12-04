import re
import logging
from typing import Any, Dict, List, Tuple
import os
import numpy as np

def tokenize_poly_input(poly_input: str): 
    # Smiles of monomers containing wildcards
    smiles = poly_input.split("|")[0]
    # Stoichiometric weights and connection rules
    weights_and_rules = "".join(poly_input.split("|", 1)[1:]).split(",")[0]
    # tokenizing smiles
    pattern = r"(\[\*\:|\[[^\]]+]|\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smi_tokens = [token for token in regex.findall(smiles)]
    assert smiles == "".join(smi_tokens), f"Tokenization mismatch. smi: {smiles}, tokens: {smi_tokens}"
    # tokenizing weights and rules
    pattern = r"([0-9]|.|\||\:|\<)"
    regex = re.compile(pattern)
    wr_tokens = [token for token in regex.findall(weights_and_rules)]
    assert weights_and_rules == "".join(wr_tokens), f"Tokenization mismatch. smi: {weights_and_rules}, tokens: {wr_tokens}"
    # list with all tokens
    tokens = smi_tokens + ["|"] + wr_tokens
    return tokens

def tokenize_poly_input_RTlike(poly_input: str): 
    """
    The tokenization of numbers in SMILES/ wildcards is different to the tokenization of float numbers. 
    Float number are tokenized similarly to procedure in Regression Transformer paper:
    https://github.com/IBM/regression-transformer/blob/main/terminator/tokenization.py
    """
    # Smiles of monomers containing wildcards
    smiles = poly_input.split("|")[0]
    # Stoichiometric weights and connection rules
    weights_and_rules = "".join(poly_input.split("|", 1)[1:]).split(",")[0]
    # tokenizing smiles
    pattern = r"(\[\*\:|\[[^\]]+]|\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smi_tokens = [token for token in regex.findall(smiles)]
    assert smiles == "".join(smi_tokens), f"Tokenization mismatch. smi: {smiles}, tokens: {smi_tokens}"
    # tokenizing weights and rules
    pattern = r"([0-9]*\.[0-9]+|-|[0-9]|\||\:|\<)"
    regex = re.compile(pattern)
    wr_tokens = [token for token in regex.findall(weights_and_rules)]
    assert weights_and_rules == "".join(wr_tokens), f"Tokenization mismatch. smi: {weights_and_rules}, tokens: {wr_tokens}"
    # list with all tokens
    pattern_float = r"([0-9]*\.[0-9])"
    regrex_float = re.compile(pattern_float)
    tokens = []
    tokens = smi_tokens + ["|"] + flatten([tokenize_float(x) if regrex_float.match(x) else x for x in wr_tokens])
    return tokens

def combine_tokens(tokens: list, tokenization="normal"):
    if tokenization=="RT_tokenized": 
        # here we need to remove the RT info about decimals again
        return ''.join([t[0] if "_" in t else t for t in tokens])
    else: 
        return ''.join(tokens)
    #TODO: unknown tokenization type error


def tokenize_float(number):
    """tokenizes float numbers, similarly to Regression Transformer paper

        Args:
            number: float

        Returns:
            list of tokens
    """
    tokens = []
    n, decimals = number.split('.')
    tokens += [f"{number}_{position}" for position, number in enumerate(n[::-1])][::-1]
    tokens += ["."]
    tokens += [f"{number}_-{position}" for position, number in enumerate(decimals, 1)]
    return tokens

def flatten(l):
    return [item for sublist in l for item in sublist]

def make_vocab(target_tokens_list: List, vocab_file: str, tokenized=True):
    assert tokenized, f"Vocab can only be made from tokenized files"

    logging.info(f"Making vocab from dataset")
    vocab = {}

    for sample in target_tokens_list:
        for token in sample:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1

    logging.info(f"Saving vocab into {vocab_file}")
    with open(vocab_file, "w") as of:
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n")
        for token, count in vocab.items():
            of.write(f"{token}\t{count}\n")


def load_vocab(vocab_file: str) -> Dict[str, int]:
    if os.path.exists(vocab_file):
        logging.info(f"Loading vocab from {vocab_file}")
    else:
        vocab_file = "./preprocessed/default_vocab_smiles.txt"
        logging.info(f"Vocab file invalid, loading default vocab from {vocab_file}")

    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            vocab[token] = i

    return vocab

def get_seq_features_from_line(tgt_tokens: List, vocab: Dict, max_tgt_len = 1024) -> Tuple[np.ndarray, int, np.ndarray, int]:

    tgt_token_ids, tgt_lens = get_token_ids(tgt_tokens, vocab, max_len=max_tgt_len)

    tgt_token_ids = np.array(tgt_token_ids, dtype=np.int32)

    return tgt_token_ids, tgt_lens

def get_token_ids(tokens: list, vocab: Dict[str, int], max_len: int) -> Tuple[List, int]:
    # token_ids = [vocab["_SOS"]]               # shouldn't really need this
    token_ids = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[:max_len-1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens

def tokenids_to_vocab(token_ids: list, vocab: Dict[str, int]):
    """Map token ids back to tokens

        Args:
            token_ids: list of token ids
            vocab: vocab as dictionary

        Returns:
            list of tokens
    """
    vocab_swap = {v: k for k, v in vocab.items()}
    return [vocab_swap[x] for x in token_ids]

def token_weights(vocab_file):

    tokens_occurences = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            try: 
                tokens_occurences[token] = int(line.strip().split("\t")[1])
            except:
                tokens_occurences[token] = 1
    coocurrance = list((dict(tokens_occurences).values()))
    symbols = list((dict(tokens_occurences).keys()))
    lamda_factor =  np.log(coocurrance)/np.sum(np.log(coocurrance))
    lamda_factor = (1/(lamda_factor+0.000001))*0.01
    weights = {}
    for i,element in enumerate(symbols):
        if lamda_factor[i] > 1.:
            lamda_factor[i] = 0.90
        weights[element] = lamda_factor[i]

    #print(weights)
    class_weights = list(weights.values())
    return class_weights

