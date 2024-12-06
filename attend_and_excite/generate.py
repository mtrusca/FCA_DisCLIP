import os
import pprint
import sys
from pathlib import Path
from typing import List
import argparse

import pyrallis
import torch
from PIL import Image
import random, json
from tqdm import tqdm

from ae.config_ae import RunConfig
from pipeline import AttendAndExcitePipeline
from ae.utils import ptp_utils, vis_utils
from ae.utils.ptp_utils import AttentionStore

from utils import parse_sentences, get_amod_dependencies, AttentionVisualizer, get_constituents
from lal_parser import KM_parser

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import spacy
from spacy.lang.en import English

def load_model():
    stable_diffusion_version = "runwayml/stable-diffusion-v1-5"
    print('stable_diffusion_version', stable_diffusion_version)
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version,
                                                     cache_dir = 'cache_dir',
                                                     torch_dtype=torch.float16).to('cuda')
    lal_parser_model_path = 'lal_parser/best_parser.pt'
    print("\nLoading lal parser model from {}...".format(lal_parser_model_path))
    assert lal_parser_model_path.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch.load(lal_parser_model_path)
    assert 'hparams' in info['spec'], "Older save files not supported"
    parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
    return stable, parser

def get_indices_to_alter(prompt, nlp1, tokenizer):
    nouns = [chunk.root.text for chunk in nlp1(prompt).noun_chunks]
    tokens = [token.text for token in tokenizer(prompt)]
    nouns_idx = [tokens.index(noun)+1 for noun in nouns]
    return nouns_idx

def run_on_prompt(prompt: List[str],
                  prompt_embeds,
                  text_inputs,
                  cross_attention_kwargs,
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    prompt_embeds=prompt_embeds,
                    text_inputs = text_inputs,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    cross_attention_kwargs=cross_attention_kwargs,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image

def run(args):
    nlp1 = spacy.load("en_core_web_sm")
    nlp2 = English()
    tokenizer = nlp2.tokenizer
    stable, parser = load_model()
    config = RunConfig()
    dataset = args.dataset
    seed = args.seed
    nb_images = args.nb_images
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    if dataset == 'AE':
        with open("data/a.e_prompts.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
    elif dataset == 'COCO':
        with open("data/COCO-10K.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
    elif dataset == 'CC':
        with open("data/CC-500.txt", "r") as f:
            lines = f.readlines()
            lines = lines[:446]
        lines = [l.strip('\n') for l in lines]
        ids = [str(i) for i in range(len(lines))]
    elif dataset == 'DAA':
        if args.type == 'all':
            with open("data/DAA-200_og.txt", "r") as f:
                lines = f.readlines()
            ids = ['{}-og'.format(i) for i in range(len(lines))]
            with open("data/DAA-200_adv.txt", "r") as f:
                lines_adv = f.readlines()
            ids_adv = ['{}-adv'.format(i) for i in range(len(lines_adv))]
            lines.extend(lines_adv)
            ids.extend(ids_adv)
            lines = [l.strip('\n') for l in lines]
        elif args.type=='adv':
            with open("data/DAA-200_adv.txt", "r") as f:
                lines = f.readlines()
            ids = ['{}-adv'.format(i) for i in range(len(lines))]
            lines = [l.strip('\n') for l in lines]
        elif args.type == 'og':
            with open("data/DAA-200_og.txt", "r") as f:
                lines = f.readlines()
            ids = ['{}-og'.format(i) for i in range(len(lines))]
            lines = [l.strip('\n') for l in lines]

    if seed >= 0:
        random.seed(seed)
    all_seeds = []
    for _ in lines:
        all_seeds.append([random.randint(0, 100000) for _ in range(nb_images)])


    for idx, prompt, p_seeds in tqdm(zip(ids, lines, all_seeds), total=len(ids)):
        print('imaginea', idx)
        untruncated_ids = stable.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        tokens = stable.tokenizer.convert_ids_to_tokens(untruncated_ids[0], skip_special_tokens=False)

        tags, tree, dep_heads, dep_labels = parse_sentences([prompt], parser)
        lal_tokens = [tag[1] for tag in tags[0]]
        dependencies, dependencies_idx = get_amod_dependencies(tags[0], dep_heads[0], dep_labels[0])
        remove_padding = False
        CLIP_variant = args.CLIP_variant
        constituents = get_constituents(tree[0], print_result=True, abstractify_high_level=True)
        for _ in range(len(constituents) - 1):
            tokens.append(tokens[-1])

        text_inputs, prompt_embeds, dependency_matrix, attribute_mask = stable._encode_prompt(
            prompt,
            num_images_per_prompt=1,
            device='cuda',
            do_classifier_free_guidance=True,
            use_focused_attention=True,
            lal_tokens=lal_tokens,
            lal_dependencies=dependencies_idx,
            used_CLIP_variant=CLIP_variant,
            DisCLIP_constituents=constituents,
            remove_padding=remove_padding
        )

        if args.use_fca == 1:
            row_idx = dependency_matrix[0].max(dim=0)[0].nonzero()
            col_idx = dependency_matrix[0].max(dim=1)[0].nonzero()
            random.seed(seed)
            random_idx = random.choice([i for i in list(range(dependency_matrix.shape[1])) if i not in row_idx])
            random_idx = 0
            for j in range(dependency_matrix.shape[1]):
                if j not in col_idx:
                    dependency_matrix[0, j, random_idx] = 10

            att_maps_seed, att_maps = None, None
            step = args.step/10
            used_kwargs_1 = {
                'fca_dependency_matrix': dependency_matrix,
                'fca_attribute_mask': attribute_mask,
                'fca_step_value': step,
                'fca_att_maps': att_maps
            }
        else:
            used_kwargs_1 = None

        for seed in p_seeds:
            old_stdout = sys.stdout
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            token_indices = get_indices_to_alter(prompt, nlp1, tokenizer)
            image = run_on_prompt(prompt=None,
                                  prompt_embeds = prompt_embeds,
                                  text_inputs = text_inputs,
                                  model=stable,
                                  controller=controller,
                                  token_indices=token_indices,
                                  cross_attention_kwargs=used_kwargs_1,
                                  seed=g,
                                  config=config)
            image.save(output_path / f'{idx}_{seed}.png')
            sys.stdout = old_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'AE')
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--seed", type=int, default=258478)
    parser.add_argument("--nb-images", type=int, default = 1)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--CLIP-variant", type=str)
    parser.add_argument("--use-fca", type=int)
    parser.add_argument("--type", type=str, default='all')
    args = parser.parse_args()
    run(args)
 