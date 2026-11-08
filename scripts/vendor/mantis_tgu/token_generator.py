# TokenGeneratorUnit extracted unchanged from vfeofanov/mantis; see NOTICE and LICENSE.
import torch
from torch import nn
from .convolution import Convolution
from .encoders import MultiScaledScalarEncoder, LinearEncoder

class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + \
            1 if patch_window_size % 2 == 0 else patch_window_size
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size,
                        out_channels=hidden_dim, dilation=1)
            for i in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (
            x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        with torch.no_grad():
            # compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = torch.mean(x_patched, axis=-1, keepdim=True)
            std_patched = torch.std(x_patched, axis=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))]

        # apply convolution for original ts and its diff
        ts_var_embeddings = []
        # diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, axis=2)
            # pad by zeros to have same dimensionality as x
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[0](self.ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # original ts
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[1](self.ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(
                embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = torch.mean(embedding, dim=2)
            patched_ts_var_embeddings.append(embedding)

        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings
