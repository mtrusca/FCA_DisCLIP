import abc
import math

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.cross_attention import CrossAttention, Attention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None,  **cross_attention_kwargs):

        if encoder_hidden_states is not None and 'fca_dependency_matrix' in cross_attention_kwargs and cross_attention_kwargs[
            'fca_dependency_matrix'] is not None:
            # print('1')
            return self.focused_attention(attn,
                                          hidden_states,
                                          cross_attention_kwargs['fca_dependency_matrix'],
                                          cross_attention_kwargs['fca_attribute_mask'],
                                          encoder_hidden_states=encoder_hidden_states,
                                          attention_mask=attention_mask,
                                          **cross_attention_kwargs)

        #h1 = self.og_attention(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)
        else:
            # print('2')
            # print('encoder_hidden_states is not None', encoder_hidden_states)
            return self.normal_attention(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)


    def og_attention(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None,  **cross_attention_kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def normal_attention(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,  **cross_attention_kwargs):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        is_cross = encoder_hidden_states is not None

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        self_attention = False
        if encoder_hidden_states is None:
            self_attention = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        del encoder_hidden_states, hidden_states
        torch.cuda.empty_cache()
        attention_scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(query.size(-1))
        attn_slice = attention_scores.softmax(dim=-1)

        self.attnstore(attn_slice.view(-1, *attn_slice.shape[2:]), is_cross, self.place_in_unet)

        if batch_size != 1:
            if 'att_vis' in cross_attention_kwargs and not self_attention:
                cross_attention_kwargs['att_vis'].add_att_maps(attn_slice.detach().cpu())

        hidden_states = torch.einsum("bhqk,bhkv->bhqv", attn_slice, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def focused_attention(self, attn: Attention, hidden_states, focused_attention_mask, w_mask,
                          encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # print('focused_attention_mask1', focused_attention_mask[0][:10,:10])
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        is_cross = encoder_hidden_states is not None

        # print('focused_attention_maskfff', focused_attention_mask.shape)
        # print('focused_attention_mask2', focused_attention_mask[0][:10, :10])

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            assert False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        attention_scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(query.size(-1))
        # print('(attention_scores.max(dim=2, keepdim=True)[0] + 1e-6)',
        #       (attention_scores.max(dim=2, keepdim=True)[0] + 1e-6)[0][0][:6])

        nb_heads = query.size(1)
        # print('focused_attention_mask3', focused_attention_mask[0][:10, :10])
        focused_attention_mask = torch.repeat_interleave(focused_attention_mask.unsqueeze(1), nb_heads, dim=1)
        # print('focused_attention_mask4', focused_attention_mask[0][0][:10, :10])
        step_value = cross_attention_kwargs['step_value'] if 'step_value' in cross_attention_kwargs else 0.6

        # print('attention_scores', attention_scores[0][0][:6, :6])
        # print('focused_attention_mask5', focused_attention_mask[0][0][:10, :10])
        focus_weights = torch.einsum("bhqk,bhdk->bhqd", attention_scores[:, :, :, :focused_attention_mask.size(-1)],
                                     focused_attention_mask )
        # print('focus_weights1', focus_weights[0][0][:6, :6])
        focus_weights = torch.abs(focus_weights)
        # print('focus_weights2', focus_weights[0][0][:6, :6])
        focus_weights = torch.where(focus_weights > 1, torch.ones_like(focus_weights),
                                        focus_weights)

        # print('focus_weights3', focus_weights[0][0][:6, :6])
        # print('(focus_weights.max(dim=2, keepdim=True)[0] + 1e-6)',
        #       (focus_weights.max(dim=2, keepdim=True)[0] + 1e-6)[0][0][:6])
        focus_weights /= (focus_weights.max(dim=2, keepdim=True)[0] + 1e-6)
        # print('focus_weights4', focus_weights[0][0][:6, :6])

        if step_value is not None:
            focus_weights = torch.where(focus_weights > step_value, torch.ones_like(focus_weights),
                                        torch.zeros_like(focus_weights))
        # print('focus_weights5', focus_weights[0][0][:6, :6])
        # if step_value is not None:
        #     focus_weights = torch.where(focus_weights > step_value, torch.ones_like(focus_weights),
        #                                 torch.zeros_like(focus_weights))
        #
        # print('focus_weights5', focus_weights[0][0][:6, :6])
        attention_scores[:, :, :, :focus_weights.size(-1)] += (focus_weights - 1) * 50

        # reweight_att_scores = True
        # if reweight_att_scores:
        #     m1 = attention_scores.amax(dim=2)[:, :, None, :focus_weights.size(-1)]
        #     m1[m1 < 0] = 0.0001
        #     attention_scores[:, :, :, :focus_weights.size(-1)] += (focus_weights - 1) * 50
        #     m2 = attention_scores.amax(dim=2)[:, :, None, :focus_weights.size(-1)]
        #     m2[m2 < 0] = 0.0001
        #     weight = m1 / m2
        #     attention_scores[:, :, :, :focus_weights.size(-1)] *= weight
        # else:
        #     attention_scores[:, :, :, :focus_weights.size(-1)] += (focus_weights - 1) * 50
        #
        #     m1 = attention_scores.amax(dim=2)[:, :, None, :focus_weights.size(-1)]
        #     m1[m1 < 0] = 0.0001
        #     attention_scores[:, :, :, :focus_weights.size(-1)] += (focus_weights - 1) * 50
        #     m2 = attention_scores.amax(dim=2)[:, :, None, :focus_weights.size(-1)]
        #     m2[m2 < 0] = 0.0001
        #     weight = m1 / m2
        #     attention_scores[:, :, :, :focus_weights.size(-1)] *= weight

        attn_slice = attention_scores.softmax(dim=-1)

        hidden_states = torch.einsum("bhqk,bhkv->bhqv", attn_slice, value)

        self.attnstore(attn_slice.view(-1, *attn_slice.shape[2:]), is_cross, self.place_in_unet)

        if batch_size != 1:
            if 'att_acc' in cross_attention_kwargs:
                cross_attention_kwargs['att_acc'].add_att_maps(attn_slice.detach())
            if 'att_vis' in cross_attention_kwargs:
                cross_attention_kwargs['att_vis'].add_att_maps(attn_slice.detach().cpu())
            if 'foc_att_vis' in cross_attention_kwargs:
                cross_attention_kwargs['foc_att_vis'].add_att_maps(focus_weights.detach().cpu())

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print('hidden_states', hidden_states)
        return hidden_states


def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # print('self.step_store', self.step_store)
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    # print('attention_maps', attention_maps)
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out