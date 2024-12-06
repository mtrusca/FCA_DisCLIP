# Object-Attribute Binding in Text-to-Image Generation: Evaluation and Control

### Summary: [Link Paper](https://arxiv.org/abs/2404.13766)

To imrpove the alignment between objects and their attributes in text-based image generation, we propose Disentangled CLIP (DisCLIP) and Focused Cross-Attention (FCA). These modules can be easily integrated into various diffusion-based image generators, improving their ability to generate multiple objects with the correct attributes as specified in text instructions.

### Attend-and-Excite

The folder ```attend_and_excite``` presents the integrations of FCA and DisCLIP into the Attend-and-Excite model. [Link Project](https://github.com/yuval-alaluf/Attend-and-Excite) | [Link Paper](https://arxiv.org/abs/2301.13826)

Installation
```
cd attend_and_excite
conda env create -f environment.yaml
conda activate fca_disclip_attend_and_excite
mv -r diffusers ./miniconda3/envs/fca_disclip_attend_and_excite/lib/python3.8/site-packages/diffusers
```
Download the LAL-parser in the folder ```lal_parser```.
```
cd lal_parser
wget https://archive.org/download/neuraladobe-ucsdparser/best_parser.pt
cd ..
```

The data is available in the folder ```data```. To generate images run: 

```
python feb_generate.py  --output-path 'results/ae'  --CLIP-variant DisCLIP --use-fca 1 --nb-images 1 --dataset AE
```

### Versatile Diffusion

The folder ```versatile_diffusion``` presents the integrations of FCA and DisCLIP into the Versatile diffusion model. [Link Project](https://github.com/SHI-Labs/Versatile-Diffusion) | [Link Paper](https://arxiv.org/abs/2211.08332)

```
cd versatile_diffusion
python -m venv fca_disclip_versatile_diffusion
source fca_disclip_versatile_diffusion/bin/activate
pip install -r requirements.txt
mv -r diffusers ./miniconda3/envs/fca_disclip_syngen/lib/python3.8/site-packages/diffusers
```
Download the pretrained model (available on the project page) and move it in ```pretrained``` folder. The data is available in the folder ```data```. To generate images run: 

```
python inference.py --dataset AE --output-path 'results/ae/vers' --CLIP-variant 1 --use-fca 1 --seed 258478
```

### SynGen

The folder ```syngen``` presents the integrations of FCA and DisCLIP into the SynGen model. [Link Project](https://github.com/RoyiRa/Linguistic-Binding-in-Diffusion-Models) | [Link Paper](https://arxiv.org/abs/2306.08877)

```
cd syngen
conda env create -f environment.yaml
conda activate fca_disclip_syngen
mv -r diffusers ./miniconda3/envs/fca_disclip_syngen/lib/python3.8/site-packages/diffusers
```

The data is available in the folder ```data```. To generate images run: 

```
python run.py --dataset AE --seed 258478 --use_focused_attention 1 --used_CLIP_variant DisCLIP --output_directory 'results/ae'

```