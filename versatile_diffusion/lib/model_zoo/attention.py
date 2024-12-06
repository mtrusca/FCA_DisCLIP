from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .diffusion_utils import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, focused_attention_mask = None, use_fca=1, mask=None):
        h = self.heads
        if context is None: use_fca = 0
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if use_fca == 1:
            step_value = 0.6
            focused_attention_mask = torch.repeat_interleave(focused_attention_mask, q.shape[0], dim=0)
            focus_weights = torch.einsum("bqk,bdk->bqd", sim[:, :, :focused_attention_mask.size(-1)],
                                         focused_attention_mask)
            focus_weights = torch.abs(focus_weights)
            focus_weights = torch.where(focus_weights > 1, torch.ones_like(focus_weights),
                                        focus_weights)
            focus_weights /= (focus_weights.max(dim=2, keepdim=True)[0] + 1e-6)
            focus_weights = torch.where(focus_weights > step_value, torch.ones_like(focus_weights),
                                        torch.zeros_like(focus_weights))
            sim[:, :, :focus_weights.size(-1)] += (focus_weights - 1) * 50

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, focused_attention_mask=None, use_fca=1):
        return checkpoint(self._forward, (x, context,
                                          focused_attention_mask, use_fca), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, focused_attention_mask=None, use_fca=1):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,
                       focused_attention_mask = focused_attention_mask, use_fca=use_fca) + x
        x = self.attn2(self.norm2(x), context=context,
                       focused_attention_mask = focused_attention_mask, use_fca=use_fca) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, focused_attention_mask=None, use_fca=1):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context, focused_attention_mask = focused_attention_mask, use_fca=use_fca)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


##########################
# transformer no context #
##########################

class BasicTransformerBlockNoContext(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, 
                                    dropout=dropout, context_dim=None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, 
                                    dropout=dropout, context_dim=None)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.checkpoint)

    def _forward(self, x):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x)) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformerNoContext(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0.,):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlockNoContext(inner_dim, n_heads, d_head, dropout=dropout)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in


#######################################
# Spatial Transformer with Two Branch #
#######################################

class DualSpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        # First crossattn
        self.norm_0 = Normalize(in_channels)
        self.proj_in_0 = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks_0 = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )
        self.proj_out_0 = zero_module(nn.Conv2d(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

        # Second crossattn
        self.norm_1 = Normalize(in_channels)
        self.proj_in_1 = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks_1 = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )
        self.proj_out_1 = zero_module(nn.Conv2d(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None, which=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        if which==0:
            norm, proj_in, blocks, proj_out = \
                self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
        elif which==1:
            norm, proj_in, blocks, proj_out = \
                self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1
        else:
            # assert False, 'DualSpatialTransformer forward with a invalid which branch!'
            # import numpy.random as npr
            # rwhich = 0 if npr.rand() < which else 1
            # context = context[rwhich]
            # if rwhich==0:
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            # elif rwhich==1:
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1

            # import numpy.random as npr
            # rwhich = 0 if npr.rand() < 0.33 else 1
            # if rwhich==0:
            #     context = context[rwhich]
            #     norm, proj_in, blocks, proj_out = \
            #         self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            # else:

            norm, proj_in, blocks, proj_out = \
                self.norm_0, self.proj_in_0, self.transformer_blocks_0, self.proj_out_0
            x0 = norm(x)
            x0 = proj_in(x0)
            x0 = rearrange(x0, 'b c h w -> b (h w) c').contiguous()
            for block in blocks:
                x0 = block(x0, context=context[0])
            x0 = rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x0 = proj_out(x0)

            norm, proj_in, blocks, proj_out = \
                self.norm_1, self.proj_in_1, self.transformer_blocks_1, self.proj_out_1
            x1 = norm(x)
            x1 = proj_in(x1)
            x1 = rearrange(x1, 'b c h w -> b (h w) c').contiguous()
            for block in blocks:
                x1 = block(x1, context=context[1])
            x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x1 = proj_out(x1)
            return x0*which + x1*(1-which) + x_in

        x = norm(x)
        x = proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = proj_out(x)
        return x + x_in