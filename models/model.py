import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
import torch.nn.functional as F
'''
1、采用更早的卷积来为attention进行局部建模
2、在建模的每一个阶段的MSA分枝上都采用一个3x3的深度卷积---
'''
def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
class DWConv(nn.Module):
    def __init__(self, dim=768,kernel=3,stride=1,padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, stride, padding, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(_init_weights)#将权重中的值全部初始化

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        #改动 dwc加上一个跳跃分枝再进行训练  dwc之后会生成一个序列
        x = self.dwconv(x, H, W) + x
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
class Attention_PVTV2(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        '''
        pvt v2 Model attention
        :param dim:
        :param num_heads:
        :param qkv_bias:
        :param qk_scale:
        :param attn_drop:
        :param proj_drop:
        :param sr_ratio:
        :param linear:
        '''
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)#自定义参数初始化方式

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):#是否是一个实例
            trunc_normal_(m.weight, std=.02)#正太分布
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)#初始化值
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)#permute 数据的位置

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class Block(nn.Module):
    def __init__(self,dim ,kernel_size=3,stride=1,padding=1, qk_scale=None,
                 mlp_ratio=4,heads=8,qkv_bias=None, drop=0.,attn_drop=0.1,drop_path=0.2,sr_ratio=1,
                 linear=False,pretrain='channels_last',layer_scale_init_value=1e-6,act_layer=False):
        '''

        :param in_channle: 输入通道 ：这里的通道是通过patch embedding后的值
        :param kernel_size:
        :param stride:
        :param padding:
        :param mlp_ratio:
        :param heads:
        :param qkv_bias:
        :param attn_drop:
        '''
        super(Block, self).__init__()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        out_channels = dim*mlp_ratio or dim
        self.MLP = MLP(in_features=dim,hidden_features=out_channels, drop=drop, linear=linear)
        self.DWC = DWConv(dim,kernel_size,stride,padding)#DWConv(dim,kernel_size,stride,padding)
        self.Attention = Attention_PVTV2(dim=dim,num_heads=heads,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.LN = LayerNorm(dim,data_format=pretrain)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        # self.conv1 = nn.Conv2d(in_channels=dim,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        # self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=dim,kernel_size=1,stride=1,padding=0)
        self.act_layer = act_layer
        if act_layer :
            self.GeLU = nn.GELU()
        #bypass branch
        self.depthcon = self.DWC
        self.apply(_init_weights)

    def forward(self,x,H,W):

        # main branch
        out = x + self.DWC(x,H,W)
        # up branch
        up_out = self.drop_path(self.LN(self.Attention(out,H,W))) + out# self.LN(self.drop_path(self.Attention(out,H,W))) + out
        if self.act_layer:
            # down_out = self.GeLU(self.DWC(out,H,W) + out)
            down_out = self.GeLU(self.DWC(out, H, W) + out)
        else:
            down_out = self.DWC(out,H,W) + out

        out = self.LN(up_out + down_out) + out# 不在使用额外的卷积 self.LN(up_out + down_out) + out
        out = out + self.LN(self.MLP(out, H, W)) #这里不再接入原始的x输入
        # out = out + x
        # out = out + self.LN(self.MLP(out,H,W))

        return out
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        '''我们不在使用torch自带的ln标准化  当输入是从convolution时候使用 channels_first   当输入是一个序列时候采用 channels_last'''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))#将一个不可训练的函数转为可训练的函数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)#给定一些ln的参数
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)#实现张量和标量之间逐元素求指数操作,或者在可广播的张量之间逐元素求指数操作

            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class MultiPhraseTransformerV1(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_chans=3,num_classes=100,embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, act_layer=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], sr_ratio=sr_ratios[i], linear=linear,act_layer=act_layer)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(_init_weights)
        # self.init_weights(pretrained)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x
#这里，Register类似于一个dict（实际上是有一个_dict属性），可以set_item和get_item。
# 关键是register函数，它可以作为装饰器，注册一个函数或者一个类
# @register_model
# def MT_v2_b0(pretrained=False, **kwargs):
#     model = MultiPhraseTransformerV1(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],linear=True,
#         **kwargs)
#     model.default_cfg = _cfg()
#     return model
#

@register_model
def MT_b0(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),linear=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

# @register_model
# def MT_b1(pretrained=False, **kwargs):
#     model = MultiPhraseTransformerV1(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()
#     #flops :2.05 GMac	params:14.03 M
#
#     return model


@register_model
def MT_b2(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def MT_b3(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    #flops :6.74 GMac	params:45.3 M
    return model


@register_model
def MT_b4(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[128, 192, 384, 640], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    #flops :10.19 GMac	params:62.06 M

    return model


@register_model
def MT_b5(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
@register_model
def MT_b2_li(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model
'''
-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------
'''


#如果此时开始选择与uniformer一样的比例
@register_model
def MT_SW_b0(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 8, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :3.47 GMac	params:24.64 M
    return model
def MT_SW_b2(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[128, 192, 448, 640], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 10, 24, 7], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :18.39 GMac	params:111.13 M  [3, 4, 18, 3]
    #flops :7.62 GMac	params:55.62 M     ----uniformer  8.3  50.3m
    return model

def MT_SW_b3(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 8, 20, 7], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :7.62 GMac	params:55.62 M     ----uniformer  8.3  50.3m
    return model


#选择和CMT一样的比例
def MT_CMT_tiny(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[46, 92, 184, 368], num_heads=[1, 2, 4, 8], mlp_ratios=[4,4,4,4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 10, 2], sr_ratios=[8, 4, 2, 1], linear=True,act_layer=True, **kwargs)
    model.default_cfg = _cfg()
    # flops :1.32 GMac	params:9.56 M
    return model
def MT_CMT_small(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[52, 104, 208, 416], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :2.16 GMac	params:15.75 M
    return model
def MT_CMT_medium(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 16, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :5.09 GMac	params:35.24 M
    return model
def MT_CMT_Large(pretrained=False, **kwargs):
    model = MultiPhraseTransformerV1(
        patch_size=4, embed_dims=[76, 152, 304, 608], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[4, 4, 20, 4], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()
    #flops :6.76 GMac	params:48.17 M
    return model
# from ptflops import get_model_complexity_info
# model =MT_b0()
# flops,params =get_model_complexity_info(model,input_res=(3,224,224),as_strings=True,print_per_layer_stat=True)
# print(f'flops :{flops}'+'\t'+f'params:{params}')

model = MT_b0()
value = torch.rand(2,3,224,224)
print(model.forward(value))