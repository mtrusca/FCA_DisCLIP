import torch, json, math, os, random, argparse
from syngen_diffusion_pipeline import SynGenDiffusionPipeline
from ast import literal_eval
from tqdm import tqdm

def main(seed, output_directory, model_path, step_size, attn_res, include_entities,
         dataset, use_focused_attention, used_CLIP_variant):
    pipe = load_model(model_path, include_entities)

    if dataset == 'AE':
        with open("/data/a.e_prompts.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
        f = open("/data/parser_ae.json")
        data = json.load(f)
    elif dataset == 'COCO':
        with open("/data/COCO-10K.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        lines = [l.strip('\n') for l in lines]
        f = open("/data/parser_coco.json")
        data = json.load(f)
    elif dataset == 'CC':
        with open("/data/CC-500.txt", "r") as f:
            lines = f.readlines()
            lines = lines[:446]
        lines = [l.strip('\n') for l in lines]
        ids = [str(i) for i in range(len(lines))]
        f = open("/data/parser_cc.json")
        data = json.load(f)
    elif dataset == 'DAA':
        with open("/data/DAA-200_og.txt", "r") as f:
            lines = f.readlines()
        ids = ['{}-og'.format(i) for i in range(len(lines))]
        with open("/data/DAA-200_adv.txt", "r") as f:
            lines_adv = f.readlines()
        ids_adv = ['{}-adv'.format(i) for i in range(len(lines_adv))]
        lines.extend(lines_adv)
        ids.extend(ids_adv)
        lines = [l.strip('\n') for l in lines]
        f = open("/data/parser_daa.json")
        data = json.load(f)

    nb_images = 1
    if seed >= 0:
        random.seed(seed)
    all_seeds = []
    for _ in lines:
        all_seeds.append([random.randint(0, 100000) for _ in range(nb_images)])

    for idx, text, p_seeds in tqdm(zip(ids, lines, all_seeds), total=len(ids)):
        print('imaginea', idx)
        dependencies_idx, constituents, lal_tokens = data[text]
        dependencies_idx = literal_eval(dependencies_idx)
        image = generate(pipe, text, p_seeds[0], step_size, attn_res,
                         use_focused_attention=use_focused_attention,
                         lal_tokens=lal_tokens,
                         lal_dependencies=dependencies_idx,
                         used_CLIP_variant=used_CLIP_variant,
                         DisCLIP_constituents=constituents
                         )
        path = output_directory + str(idx) + "_" + str(p_seeds[0]) + '.png'
        image.save(path)


def load_model(model_path, include_entities):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline.from_pretrained(model_path,
                                                   include_entities=include_entities,
                                                   cache_dir = 'cache_dir').to(device)

    return pipe


def generate(pipe, prompt, seed, step_size, attn_res, use_focused_attention=0,
                         lal_tokens=None,
                         lal_dependencies=None,
                         used_CLIP_variant='CLIP',
                         DisCLIP_constituents=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    result = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size,
                  attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
                  use_focused_attention=use_focused_attention,
                  lal_tokens=lal_tokens,
                  lal_dependencies=lal_dependencies,
                  used_CLIP_variant=used_CLIP_variant,
                  DisCLIP_constituents=DisCLIP_constituents
                  )
    return result['images'][0]


def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{prompt}_{seed}.png"
    image.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=1924
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='./output'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=20.0,
        help='The SynGen step size'
    )

    parser.add_argument(
        '--attn_res',
        type=int,
        default=256,
        help='The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)'
    )

    parser.add_argument(
        '--include_entities',
        type=bool,
        default=False,
        help='Apply negative-only loss for entities with no modifiers'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='AE',
    )
    parser.add_argument(
        '--use_focused_attention',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--used_CLIP_variant',
        type=str,
        default='DisCLIP',
    )
    args = parser.parse_args()
    main(args.seed, args.output_directory, args.model_path, args.step_size, args.attn_res,
         args.include_entities, args.dataset, args.use_focused_attention, args.used_CLIP_variant)

