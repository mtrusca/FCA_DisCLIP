import numpy as np
import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim import DDIMSampler
from utils import adjust_rank
import argparse, random, json
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm

n_samples  = 2
n_sample_text  = 4
cache_examples = True
n_sample_image = 2
scale_textto = 7.5
h, w = [512, 512]
ddim_steps = 50
ddim_eta = 0.0
scale = 7.5
image_latent_dim = 4
text_latent_dim = 768
text_temperature = 1
scale_imgto = 7.5
disentanglement_noglobal = True
dtype = torch.float16

def run(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    dataset = args.dataset
    seed = args.seed
    use_disclip, use_fca = args.CLIP_variant, args.use_fca
    if dataset == 'AE':
        with open("data/a.e_prompts.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
        f = open("data/parser_ae.json")
        data = json.load(f)
    elif dataset == 'COCO':
        with open("data/COCO-10K.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
        f = open("data/parser_coco.json")
        data = json.load(f)
    elif dataset == 'CC':
        with open("data/CC-500.txt", "r") as f:
            lines = f.readlines()
            lines = lines[:446]
        lines = [l.strip('\n') for l in lines]
        ids = [str(i) for i in range(len(lines))]
        f = open("data/parser_cc.json")
        data = json.load(f)
    elif dataset == 'DAA':
        with open("data/DAA-200_og.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        with open("data/DAA-200_adv.txt", "r") as f:
            lines_adv = f.readlines()
        ids_adv = ['{}-adv'.format(i) for i in range(len(lines_adv))]
        lines.extend(lines_adv)
        ids.extend(ids_adv)
        lines = [l.strip('\n') for l in lines]
        f = open("data/parser_daa.json")
        data = json.load(f)

    if seed >= 0:
        random.seed(seed)
    all_seeds = []
    for _ in lines:
        all_seeds.append([random.randint(0, 100000) for _ in range(args.nb_images)])

    cfgm = model_cfg_bank()('vd_four_flow_v1-0')
    net = get_model()(cfgm)
    net.ctx['text'].fp16 = True
    net.ctx['image'].fp16 = True
    net = net.half()
    sd = torch.load('pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
    net.load_state_dict(sd, strict=False)
    net.to('cuda')
    sampler = DDIMSampler(net)
    adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)

    q = 0
    for idx, text, p_seeds in tqdm(zip(ids, lines, all_seeds), total=len(ids)):
        print('imaginea', idx)
        dependencies_idx, constituents, lal_tokens = data[text]
        dependencies_idx = literal_eval(dependencies_idx)
        print(text)
        print(data[text])

        u, _ = net.ctx_encode([""], which='text', constituents = constituents, use_disclip = 0,
                           lal_dependencies = dependencies_idx, lal_tokens = lal_tokens,
                           n_samples = 1, use_fca = 0, seed = seed)
        u = u.repeat(n_samples, 1, 1)
        c, dependency_matrix = net.ctx_encode([text], which='text', constituents = constituents, use_disclip = use_disclip,
                           lal_dependencies=dependencies_idx, lal_tokens=lal_tokens,
                           n_samples=1, use_fca=use_fca, seed = seed)
        c = c.repeat(n_samples, 1, 1)
        print('dependency_matrix', dependency_matrix.shape)

        shape = [n_samples, image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=ddim_steps,
            x_info={'type': 'image'},
            c_info={'type': 'text', 'conditioning': c, 'unconditional_conditioning': u,
                    'unconditional_guidance_scale': scale},
            shape=shape,
            verbose=False,
            dependency_matrix = dependency_matrix,
            use_fca = use_fca,
            eta=ddim_eta)
        im = net.vae_decode(x, which='image')
        im = [tvtrans.ToPILImage()(i) for i in im]
        im[0].save(output_path / f'{idx}_{seed}.png')
        im[0].save('test1.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'AE')
    parser.add_argument("--output-path", type=str, default='')
    parser.add_argument("--seed", type=int, default=258478)
    parser.add_argument("--CLIP-variant", type=int)
    parser.add_argument("--nb-images", type=int, default=1)
    parser.add_argument("--use-fca", type=int)
    args = parser.parse_args()
    run(args)
