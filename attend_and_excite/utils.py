import torch
from lal_parser import KM_parser
from nltk import word_tokenize, sent_tokenize
import nltk
import torch.nn.functional as F
from tqdm import tqdm
import math
import re
REVERSE_TOKEN_MAPPING = dict([(value, key) for key, value in KM_parser.BERT_TOKEN_MAPPING.items()])


# This class is used for saving and averaging the attention maps
class AttentionVisualizer():
    def __init__(self, prompt, nb_layers):
        self.attention_maps = {} # layer : tensor[batch size (uncond + cond) | heads | positions (w x h) | keys (text embeddings (77) + positions)]
        self.curr_layer = 0
        self.curr_step = 0
        self.prompt = prompt
        self.nb_layers = nb_layers

    def add_att_maps(self, x):
        self.attention_maps.setdefault(self.curr_layer, []).append(x)
        self.curr_layer += 1
        if self.curr_layer == self.nb_layers:
            self.next_step()

    def next_step(self):
        self.curr_layer = 0
        self.curr_step += 1

    def get_attention_maps(self):
        return self.attention_maps

    def get_attention_maps(self, layers=None, steps=None, heads=None, size=64):
        if isinstance(layers, int):
            layers = [layers]
        if isinstance(steps, int):
            steps = [steps]
        if isinstance(heads, int):
            heads = [heads]
        if layers is None:
            layers = range(len(self.attention_maps))

        maps = torch.tensor([]) # tensor[batch size (uncond + cond) | heads | steps | layers | words | w | h]
        for layer_idx in layers:
            layer_maps = torch.stack(self.attention_maps[layer_idx], dim=2).transpose(-1,-2)[:,:,:,:len(self.prompt)]
            d = round(math.sqrt(layer_maps.size(-1)))
            layer_maps = layer_maps.view(layer_maps.size(0), layer_maps.size(1), layer_maps.size(2), layer_maps.size(3), d, d)

            layer_maps = layer_maps[1]
            if heads is not None:
                layer_maps = layer_maps[heads]
            layer_maps = layer_maps.mean(dim=0)
            if steps is not None:
                layer_maps = layer_maps[steps]
            layer_maps = layer_maps.mean(dim=0)

            layer_maps = layer_maps.unsqueeze(0)
            layer_maps = F.interpolate(layer_maps.to(torch.float32), size=(size, size), mode='bicubic')
            maps = torch.cat([maps, layer_maps], dim=0)

        maps = maps.mean(dim=0)
        return maps



# this is a helper function of of the function get_constituents
def _iterate_over_const(match, constituents, abstractify_high_level):
    split = match.group(1).split(' ')
    det = split[0]
    sent_part = ' '.join(split[1:])
    if det[:2] == 'NN':
        sent_part = '[' + sent_part + ']'
    if det == 'NP' or det == 'NX':
        nns = re.findall(r'\[([^[]*)\]', sent_part)
        const, _ = re.subn(r'\[([^[]*)\]', lambda m: m.group(1), sent_part)
        constituents.append((const, nns))
        if abstractify_high_level and len(nns)==1:
            sent_part = '[' + nns[0] + ']'
    return sent_part

# this function is used to obtain the constituents of an abstracted constituency tree, given a constituency tree
def get_constituents(const_tree, print_result=True, abstractify_high_level=True):
    constituents = []
    text = const_tree
    n = 1
    while n:
        text, n = re.subn(r'\(([^()]*)\)', lambda m: _iterate_over_const(m, constituents, abstractify_high_level), text)
    if print_result:
        print('found constituents and their dependency:')
        for const in constituents:
            print('\t {} -> {}'.format(*const))
    return constituents

# this function is used to obtain the attribute dependencies given a dependency parse
def get_amod_dependencies(tagged_sentence, heads, labels, print_result=True):
    dependencies = []
    dependencies_idx = []
    for i, ((_, word), head, label) in enumerate(zip(tagged_sentence, heads, labels)):
        if label == 'amod' and head != 0:
            dependencies.append((word, tagged_sentence[head-1][1]))
            dependencies_idx.append((i, head - 1))
    if print_result:
        print('found dependencies:')
        for dep in dependencies:
            print('\t {} -> {}'.format(*dep))
    return dependencies, dependencies_idx

# This code uses the LAL parser to parse a sentence into a dependency- and constituency tree
def parse_sentences(sentences,
                    # model_path_base='./LAL-Parser/best_parser.pt',
                    parser,
                    contributions=False,
                    max_tokens=-1,
                    pos_tag=1,
                    batch_size=50):
    # print("\nLoading lal parser model from {}...".format(model_path_base))
    # assert model_path_base.endswith(".pt"), "Only pytorch savefiles supported"
    #
    # info = torch.load(model_path_base)
    # assert 'hparams' in info['spec'], "Older save files not supported"
    # parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
    parser.contributions = contributions
    parser.eval()
    print("\nParsing sentences...")

    if max_tokens > 0:
        tmp = []
        for sentence in sentences:
            sub_sentences = [word_tokenize(sub_sentence) for sub_sentence in sent_tokenize(sentence)]
            this_sentence = sub_sentences[0][:max_tokens]
            this_idx = 1
            move_on = False
            while this_idx < len(sub_sentences) and len(this_sentence) < max_tokens and not move_on:
                if len(sub_sentences[this_idx]) <= max_tokens - len(this_sentence):
                    this_sentence = this_sentence + sub_sentences[this_idx]
                else:
                    move_on = True
                this_idx += 1
            tmp.append(' '.join(this_sentence))
        sentences = tmp

    if pos_tag == 2:
        # Parser does not do tagging, so use a dummy tag when parsing from raw text
        if 'UNK' in parser.tag_vocab.indices:
            dummy_tag = 'UNK'
        else:
            dummy_tag = parser.tag_vocab.value(0)

    syntree_pred = []
    for start_index in tqdm(range(0, len(sentences), batch_size), desc='Parsing sentences'):
        subbatch_sentences = sentences[start_index:start_index + batch_size]
        if pos_tag == 2:
            tagged_sentences = [[(dummy_tag, REVERSE_TOKEN_MAPPING.get(word, word)) for word in word_tokenize(sentence)]
                                for sentence in subbatch_sentences]
        elif pos_tag == 1:
            tagged_sentences = [
                [(REVERSE_TOKEN_MAPPING.get(tag, tag), REVERSE_TOKEN_MAPPING.get(word, word)) for word, tag in
                 nltk.pos_tag(word_tokenize(sentence))] for sentence in subbatch_sentences]
        else:
            tagged_sentences = [[(REVERSE_TOKEN_MAPPING.get(word.split('_')[0], word.split('_')[0]),
                                  REVERSE_TOKEN_MAPPING.get(word.split('_')[1], word.split('_')[1])) for word in
                                 sentence.split()] for sentence in subbatch_sentences]
        syntree, _ = parser.parse_batch(tagged_sentences)
        syntree_pred.extend(syntree)

    res_dep_head = [[leaf.father for leaf in tree.leaves()] for tree in syntree_pred]
    res_dep_type = [[leaf.type for leaf in tree.leaves()] for tree in syntree_pred]
    res_const_trees = []
    for tree in syntree_pred:
        res_const_trees.append("{}\n".format(tree.convert().linearize()))
    return tagged_sentences, res_const_trees, res_dep_head, res_dep_type
