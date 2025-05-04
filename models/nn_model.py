import torch
import torch.nn as nn
from torchviz import make_dot
import models.hyperparams as HP


# TODO: Всё правильно и проверено
class CrossAttentionMultiHead(nn.Module):
    def __init__(self, text_dim, img_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (img_dim // num_heads) ** -0.5
        # Проекции для изображения
        self.to_q = nn.Linear(img_dim, img_dim)
        # Проекции для текста
        self.to_kv = nn.Linear(text_dim, img_dim * 2)
        # Нормализация
        self.norm = nn.LayerNorm(img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_emb, mask=None):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, H*W, C]
        # Query из изображения
        q = self.to_q(x_flat)
        # Key/Value из текста
        k, v = self.to_kv(text_emb).chunk(2, dim=-1)
        # Multi-head attention
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, H*W, C//nh]
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, L, C//nh]
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, nh, L, C//nh]
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, H*W, L]
        # Применение маски (если она есть)
        if mask is not None:
            # mask: [B, L] → [B, 1, 1, L] (подходит для broadcating)
            mask = mask[:, None, None, :].to(attn.dtype)
            attn = attn.masked_fill(mask == 0, float('-inf'))  # -inf → softmax → 0
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        # out = x_flat + 1.0 * out  # Малый вес для стабильности (0,5 пока подходит)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 8, num_heads: int = 4):
        super(SelfAttentionBlock, self).__init__()
        # GroupNorm
        if num_channels % num_groups != 0:
            num_groups = self.compute_groups(num_channels, num_groups)
        self.norm = nn.GroupNorm(num_groups, num_channels)
        # Self-Attention
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads,
            batch_first=True,
            dropout=0.1
        )

    def compute_groups(self, channels, max_groups=8):
        for groups in range(max_groups, 0, -1):
            if channels % groups == 0:
                return groups
        return 1  # Если ничего не найдено

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self,
                 n_out: int,  # Output Dimension
                 t_emb_dim: int  # Time Embedding Dimension
                 ):
        super(TimeEmbedding, self).__init__()
        # Time Embedding Block
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, batch_size):
        super().__init__()
        self.bs = batch_size
        self.te_block = TimeEmbedding(in_channels, time_emb_dim)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                stride=1)
        if self.bs < 16:
            if in_channels % 8 != 0:
                self.norm_1 = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, affine=True)
            else:
                self.norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=True)
            self.norm_2 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
        else:
            self.norm_1 = nn.BatchNorm2d(num_features=in_channels)
            self.norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()
        self.dropout = nn.Dropout(p=0.1)  # 10% нейронов будут обнулены

    def forward(self, x, time_emb, log_D_proj=None):
        if log_D_proj != None:
            x = x + log_D_proj
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)
        r = self.dropout(r)
        r = self.norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)
        r = r + self.residual(x)
        return r


class ResNetMiddleBlock(ResNetBlock):
    def __init__(self, in_channels, out_channels, time_emb_dim, batch_size):
        super().__init__(in_channels, out_channels, time_emb_dim, batch_size)
        self.self_attention = SelfAttentionBlock(out_channels)

    def forward(self, x, time_emb, log_D_proj=None):
        if log_D_proj != None:
            x = x + log_D_proj
        x = x + self.te_block(time_emb)[:, :, None, None]
        r = self.norm_1(x)
        r = self.silu_1(r)
        r = self.conv_1(r)
        r = self.self_attention(r)
        r = self.dropout(r)
        r = self.norm_2(r)
        r = self.silu_2(r)
        r = self.conv_2(r)
        r = r + self.residual(x)
        return r


class DeepBottleneck(nn.Module):
    def __init__(self, in_C, out_C, text_emb_dim, time_emb_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.deep_block_1 = ResNetBlock(in_C, out_C, time_emb_dim, self.batch_size)
        self.cross_attn_multi_head_1 = CrossAttentionMultiHead(text_emb_dim, out_C)

    # здесь правильный порядок преобразований!!!
    def forward(self, x, text_emb, time_emb, attention_mask, log_D_proj):
        x = self.deep_block_1(x, time_emb, log_D_proj)
        if text_emb != None:
            x = self.cross_attn_multi_head_1(x, text_emb, attention_mask)
        return x


class SoftUpsample(nn.Module):
    def __init__(self, in_C, out_C):
        super().__init__()
        # Увеличиваем размер изображения в 2 раза, можем уменьшить кол-во каналов
        self.very_soft_up = nn.Sequential(
            # Более мягкая версия увеличения в 2 раза, помогающая убрать шахматные артефакты
            nn.ConvTranspose2d(in_C, out_C, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # Сглаживающая свёртка
            nn.Conv2d(out_C, out_C, kernel_size=3, padding=1, stride=1),
            # Максимальная мягкость
            nn.SiLU()
        )

    def forward(self, x):
        x = self.very_soft_up(x)
        return x


class MyUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.txt_emb_dim = self.config['TEXT_EMB_DIM']
        self.time_emb_dim = self.config['TIME_EMB_DIM']
        self.batch_size = self.config['BATCH_SIZE']
        self.orig_img_channels = self.config['ORIG_C']

        first_out_C = self.config['DOWN'][0]['in_C']

        self.prepare = nn.Conv2d(self.orig_img_channels, first_out_C, kernel_size=1, padding=0,
                                 stride=1)  # Не меняем изображение, просто подготавливаем его для дальнейшей обработки, увеличивая кол-во каналов (линейное преобразование)
        self.prepare_with_log_D_proj =  nn.Conv2d(self.orig_img_channels + 1, first_out_C, kernel_size=1,
                                                                padding=0,
                                                                stride=1)
        self.down_blocks = nn.ModuleList()
        down_blocks = self.config['DOWN']
        for db in down_blocks:
            self.down_blocks.append(
                self.create_down_block(db['in_C'], db['in_C'], self.time_emb_dim, self.batch_size, db['SA']))
            self.down_blocks.append(
                nn.Conv2d(db['in_C'], db['out_C'], kernel_size=3, padding=1, stride=2))  # уменьшение размера в 2 раза

        self.bottleneck = nn.ModuleList()
        bottleneck = self.config['BOTTLENECK']
        for bn in bottleneck:
            self.bottleneck.append(
                self.create_bottleneck_block(bn['in_C'], bn['out_C'], self.txt_emb_dim, self.time_emb_dim,
                                             self.batch_size))

        self.up_blocks = nn.ModuleList()
        up_blocks = self.config['UP']
        for ub in up_blocks:
            self.up_blocks.append(SoftUpsample(ub['in_C'], ub['out_C']))
            self.up_blocks.append(
                self.create_up_block(ub['out_C'] + ub['sc_C'], ub['out_C'] + ub['sc_C'], self.time_emb_dim,
                                     self.batch_size,
                                     ub['SA']))
            if ub['CA']:
                self.up_blocks.append(CrossAttentionMultiHead(self.txt_emb_dim, ub['out_C'] + ub['sc_C']))

        in_C_final = self.config['UP'][-1]['out_C'] + self.config['UP'][-1]['sc_C']
        out_C_final = (in_C_final // 2) - first_out_C
        self.up_final = SoftUpsample(in_C_final, out_C_final)
        in_C = (first_out_C + out_C_final)
        out_C = in_C // 2
        self.single_conv_final = nn.Conv2d(in_C,
                                           out_C,
                                           kernel_size=3, padding=1,
                                           stride=1)
        in_C = out_C
        self.final = nn.Sequential(
            nn.Conv2d(in_C, self.orig_img_channels, kernel_size=3, padding=1, stride=1),
        )  # В ddpm без финальной активации

    def create_down_block(self, in_channels, out_channels, time_emb_dim, batch_size, use_self_attn):
        if not use_self_attn:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)
        else:
            return ResNetMiddleBlock(in_channels, out_channels, time_emb_dim, batch_size)

    def create_bottleneck_block(self, in_channels, out_channels, text_emb_dim, time_emb_dim, batch_size):
        return DeepBottleneck(in_channels, out_channels, text_emb_dim, time_emb_dim, batch_size)

    def create_up_block(self, in_channels, out_channels, time_emb_dim, batch_size,
                        use_self_attn):
        if use_self_attn:
            return ResNetMiddleBlock(in_channels, out_channels, time_emb_dim, batch_size)
        else:
            return ResNetBlock(in_channels, out_channels, time_emb_dim, batch_size)

    def forward(self, x, text_emb, time_emb, attension_mask, log_D_proj=None):
        if log_D_proj != None:
            x = torch.cat([x, log_D_proj], dim=1)  # x: [B, C, H, W], log_D: [B, 1, H, W] -> [B, C+1, H, W]
            x = self.prepare_with_log_D_proj(x)
        else:
            x = self.prepare(x)
        skip_connections = []
        # Энкодер (downsampling)
        for i in range(0, len(self.down_blocks), 2):
            down = self.down_blocks[i]
            x = down(x, time_emb, log_D_proj)
            skip_connections.append(x)
            downsample = self.down_blocks[i + 1]
            x = downsample(x)
        # Bottleneck
        for bn in self.bottleneck:
            x = bn(x, text_emb, time_emb, attension_mask, log_D_proj)
        # if HP.VIZ_STEP:
        #     HP.VIZ_STEP = False
        #     visualize_tsne(x)
        # Декодер (upsampling)
        i = 0
        for decoder in self.up_blocks:
            if isinstance(decoder, SoftUpsample):
                x = decoder(x)
                skip = skip_connections[-(i + 1)]
                x = torch.cat([x, skip], dim=1)  # Skip connection
                i += 1
            elif isinstance(decoder, ResNetBlock) or isinstance(decoder, ResNetMiddleBlock):
                x = decoder(x, time_emb, log_D_proj)
            elif isinstance(decoder, CrossAttentionMultiHead):
                x = decoder(x, text_emb, attension_mask)
        x = self.up_final(x)
        skip = skip_connections[0]
        x = torch.cat([x, skip], dim=1)
        x = self.single_conv_final(x)
        x = self.final(x)
        return x

# # Адаптер-модель для визуализации
# class WrappedModel(nn.Module):
#     def __init__(self, model, text_emb, time_emb, attn_mask):
#         super().__init__()
#         self.model = model
#         self.time_emb = time_emb
#         self.text_emb = text_emb
#         self.attn_mask = attn_mask
#
#     def forward(self, x):
#         return self.model(x, self.text_emb, self.time_emb, self.attn_mask)


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
#
# # Пример: latent_vectors — torch.Tensor [B, D], labels — torch.Tensor [B]
# def visualize_tsne(bottleneck_output):
#     latent = bottleneck_output  # размер [B, C, H, W]
#     B = latent.shape[0]
#     flattened_latent = latent.reshape(B, -1)  # → [B, C*H*W]
#
#     tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, n_iter=1000)
#     latent_2d = tsne.fit_transform(flattened_latent.cpu().detach().numpy())
#
#     # labels — это ground truth метки для анализа кластеров
#     plt.figure(figsize=(8, 6))
#     plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='red', cmap='tab10', alpha=0.7)
#     plt.colorbar()
#     plt.title("t-SNE визуализация латентного пространства")
#     plt.savefig("trained/latent_dim/tsne_latent_dim_label_0.png", dpi=300)  # dpi — качество (точек на дюйм)
#     plt.close()
#     # plt.show()
#     # plt.pause(3600)


# # Переводим в numpy
# if isinstance(latent_vectors, torch.Tensor):
#     latent_vectors = latent_vectors.detach().cpu().numpy()
# if isinstance(labels, torch.Tensor):
#     labels = labels.cpu().numpy()
#
# # t-SNE
# tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
# reduced = tsne.fit_transform(latent_vectors)
#
# # Визуализация
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
# plt.title("t-SNE visualization of latent space")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.legend(title="Class")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
