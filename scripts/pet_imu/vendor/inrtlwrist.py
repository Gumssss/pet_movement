"""Released InrtlWrist encoder classes from SPICExLAB/IMU-Bench (MIT).
Source: https://github.com/SPICExLAB/IMU-Bench/blob/main/imu_lm/models/ViT1D/model.py
Upstream file SHA-256: 7fe3872082f639c5bc569af4b085c029c7fcb376704bbfa22b21a2a92a61c753
License: LICENSE.InrtlWrist. Only the config accessor is adapted for a plain dictionary.
"""
from typing import Any

import torch
from torch import nn


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding: [B, C, T] → [B, N, D] where N = C * (T // patch_size).

    shared_kernel modes
    -------------------
    True        : one Conv1d(1→D) shared across all channels (LSM-2 style)
    "acc_gyro"  : two Conv1ds — proj_acc for ch0-2 (accelerometer),
                                proj_gyro for ch3-5 (gyroscope)
    False       : one Conv1d(C→D) — all channels projected jointly

    input_transform
    ---------------
    "none" (default) : each patch is the raw `patch_size` time samples.
    "fft"            : each patch is its own complex spectrum, real and imaginary
                       parts interleaved — FOCAL's native input representation
                       (Liu et al. 2023, src/data_augmenter/Augmenter.py::fft_preprocess:
                       ``torch.fft.fft(x, dim=-1)`` over the samples of one sub-interval,
                       then ``torch.view_as_real``). Here the sub-interval IS the patch,
                       so the projection kernel widens from patch_size to 2*patch_size
                       and the token count is unchanged. FOCAL stacks real/imag along the
                       channel axis of a DeepSense conv; we hand the same 2*patch_size
                       numbers to the per-patch projection, the faithful analogue for a
                       patch-wise linear stem.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        shared_kernel=True,   # bool or "acc_gyro"
        input_transform: str = "none",
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.embed_dim   = embed_dim
        self.shared_kernel = shared_kernel
        self.input_transform = str(input_transform or "none")
        if self.input_transform not in ("none", "fft"):
            raise ValueError(f"PatchEmbed1D: unknown input_transform {input_transform!r}")
        # fft widens each patch from patch_size real samples to patch_size complex bins
        # stored as (real, imag) pairs -- 2*patch_size numbers, still exactly one token.
        kern = patch_size * (2 if self.input_transform == "fft" else 1)
        self.kernel_size = kern

        if shared_kernel == "acc_gyro":
            # Separate kernels for accelerometer (ch0-2) and gyroscope (ch3-5)
            self.proj_acc  = nn.Conv1d(1, embed_dim, kernel_size=kern, stride=kern)
            self.proj_gyro = nn.Conv1d(1, embed_dim, kernel_size=kern, stride=kern)
        elif shared_kernel:
            # Single Conv1d applied to each channel independently
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=kern, stride=kern)
        else:
            # All channels projected jointly
            self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=kern, stride=kern)

    def _fft_expand(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C, 2T]: per-patch FFT, real/imag interleaved.

        Same operation as FOCAL's fft_preprocess: FFT along the samples of one
        sub-interval (here: one patch), then view_as_real. Patches are contiguous and
        non-overlapping, matching the Conv1d(kernel=stride) patching of the time-domain
        stem, so patch p of the output feeds exactly the projection position that patch p
        of the time-domain input would.
        """
        B, C, T = x.shape
        ps = self.patch_size
        Pt = T // ps
        xp = x[..., : Pt * ps].reshape(B, C, Pt, ps)
        f = torch.fft.fft(xp.float(), dim=-1)          # [B, C, Pt, ps] complex
        f = torch.view_as_real(f)                      # [B, C, Pt, ps, 2]
        return f.reshape(B, C, Pt * 2 * ps).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C*num_patches, D]."""
        B, C, T = x.shape
        if self.input_transform == "fft":
            x = self._fft_expand(x)

        if self.shared_kernel == "acc_gyro":
            # ch0-2 → proj_acc, ch3-5 → proj_gyro
            n_acc = C // 2   # 3 acc channels, 3 gyro channels
            patches = []
            for c in range(C):
                proj = self.proj_acc if c < n_acc else self.proj_gyro
                patches.append(proj(x[:, c:c+1, :]).transpose(1, 2))  # [B, Pt, D]
            return torch.cat(patches, dim=1)   # [B, C*Pt, D]
        elif self.shared_kernel:
            patches = []
            for c in range(C):
                patches.append(self.proj(x[:, c:c+1, :]).transpose(1, 2))  # [B, Pt, D]
            return torch.cat(patches, dim=1)   # [B, C*Pt, D]
        else:
            return self.proj(x).transpose(1, 2)


class PositionalEmbedding2D(nn.Module):
    """2D positional embedding: separate embeddings for time and channel positions."""
    
    def __init__(
        self,
        max_patches_per_channel: int,
        max_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(max_patches_per_channel, embed_dim)
        self.channel_embed = nn.Embedding(max_channels, embed_dim)
        self.max_patches_per_channel = max_patches_per_channel
        self.max_channels = max_channels
    
    def forward(self, num_patches_per_channel: int, num_channels: int, device: torch.device) -> torch.Tensor:
        """Generate positional embeddings for [N] tokens where N = num_channels * num_patches_per_channel."""
        # Time positions: [0, 1, 2, ..., num_patches-1] repeated for each channel
        time_pos = torch.arange(num_patches_per_channel, device=device)
        time_pos = time_pos.repeat(num_channels)  # [N]
        
        # Channel positions: [0, 0, ..., 1, 1, ..., C-1, C-1, ...]
        channel_pos = torch.arange(num_channels, device=device)
        channel_pos = channel_pos.repeat_interleave(num_patches_per_channel)  # [N]
        
        # Sum embeddings
        pos_embed = self.time_embed(time_pos) + self.channel_embed(channel_pos)  # [N, D]
        return pos_embed


class DropPath(nn.Module):
    """Stochastic depth: drop the entire residual branch with probability p."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # BUG FIX (2026-08-15): this used to be
        #   torch.rand(shape).floor_() + keep_prob
        # i.e. floor(U[0,1)) == 0 always, so mask == keep_prob for every
        # sample and x * keep_prob / keep_prob == x -- DropPath was a silent
        # no-op for every drop_path_rate > 0 ever configured in this repo
        # (18 configs, all standard-ViT1D BEiT/DINO/HAR-finetune runs; the
        # alt-attn encoder never wired drop_path_rate at all until today).
        # Correct timm/MAE form: floor(U[0,1) + keep_prob) is 1 w.p. keep_prob.
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob).floor_()
        return x * mask / keep_prob


class _PreNormBlock(nn.Module):
    """Standard pre-norm transformer block: x + Attn(LN x), x + FFN(LN x).

    drop_path (2026-08-15): optional stochastic depth on both residual
    branches (no parameters -> state_dicts are unchanged; default 0.0 keeps
    every existing checkpoint's behaviour bit-identical)."""

    def __init__(self, d_model, nhead, dim_ff, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):                 # [*, N, D]
        n = self.norm1(x)
        x = x + self.drop_path(self.attn(n, n, n)[0])
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class ViT1DEncoderAltAttn(nn.Module):
    """ViT-1D encoder with alternating frame-wise / global attention (VGGT-style).

    A "frame" (0.32 s section) is one time-patch position across all C channels:
    its C tokens attend to each other in *frame-wise* attention. *Global*
    attention then lets every token (all sections × channels) plus a learnable
    global CLS token attend to one another. Layers alternate frame → global.
    The CLS token participates only in global attention (excluded from frame
    attention). Uses BERT-style masking so the section grid stays intact.

    Token grid: [B, Pt, C, D]  (Pt = T // patch_size sections, C channels).
    Variable length is supported natively (Pt varies per batch).
    """

    def __init__(self, cfg: Any):
        super().__init__()
        vit_cfg  = cfg.get("vit1d", {}) or {}
        enc_cfg  = vit_cfg.get("encoder", {}) or {}
        data_cfg = cfg.get("data", {}) or {}

        sensor_cols = data_cfg.get("sensor_columns", ["acc_x", "acc_y", "acc_z"])
        self.in_channels = len(sensor_cols)
        self.patch_size  = int(vit_cfg.get("patch_size", 16))
        _sk = vit_cfg.get("shared_kernel", True)
        shared_kernel = _sk if isinstance(_sk, str) else bool(_sk)

        self.embed_dim = int(enc_cfg.get("hidden_size", 384))
        num_layers     = int(enc_cfg.get("num_hidden_layers", 12))
        num_heads      = int(enc_cfg.get("num_attention_heads", 6))
        dim_ff         = int(enc_cfg.get("intermediate_size", 1536))
        dropout        = float(enc_cfg.get("hidden_dropout_prob", 0.0))
        max_patches    = int(vit_cfg.get("max_patches_per_channel", 256))
        max_channels   = int(vit_cfg.get("max_channels", 32))

        self.input_transform = str(vit_cfg.get("input_transform", "none") or "none")
        self.patch_embed = PatchEmbed1D(self.in_channels, self.patch_size,
                                        self.embed_dim, shared_kernel=shared_kernel,
                                        input_transform=self.input_transform)
        self.pos_embed   = PositionalEmbedding2D(max_patches, max_channels, self.embed_dim)

        # Global CLS token (global attention only) + BERT mask token.
        # `use_cls: false` removes the CLS entirely: global attention then runs over
        # the channel-time tokens alone and the pooled read-out becomes a mean over
        # tokens. Default true, so every existing config and checkpoint is unchanged.
        self._use_cls = bool(enc_cfg.get("use_cls", True))
        if self._use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Optional per-section register token (VGGT-style): one learnable register
        # per 0.32 s section, participates in BOTH frame and global attention and is
        # pretrained self-supervised (it aggregates section info that helps the MAE
        # decoder reconstruct masked patches via global attention). Downstream pose
        # heads can read off these pretrained section features.
        self._use_register = bool(enc_cfg.get("use_register", False))
        if self._use_register:
            self.register_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.register_pos   = nn.Embedding(max_patches, self.embed_dim)  # per-section id
            self.register_norm  = nn.LayerNorm(self.embed_dim)
            nn.init.trunc_normal_(self.register_token, std=0.02)

        # Alternating blocks: frame, global, frame, global, ...
        # drop_path_rate (2026-08-15): stochastic depth, linearly increasing
        # from 0 at the first pair to drop_path_rate at the last pair (MAE
        # convention); default 0.0 = off (previously silently ignored here).
        n_pairs = num_layers // 2
        drop_path_rate = float(enc_cfg.get("drop_path_rate", 0.0))
        dpr = [drop_path_rate * i / max(1, n_pairs - 1) for i in range(n_pairs)]
        self.frame_blocks  = nn.ModuleList(
            [_PreNormBlock(self.embed_dim, num_heads, dim_ff, dropout, drop_path=dpr[i]) for i in range(n_pairs)])
        self.global_blocks = nn.ModuleList(
            [_PreNormBlock(self.embed_dim, num_heads, dim_ff, dropout, drop_path=dpr[i]) for i in range(n_pairs)])
        self.n_pairs = n_pairs

        self.norm     = nn.LayerNorm(self.embed_dim)
        if self._use_cls:
            self.cls_norm = nn.LayerNorm(self.embed_dim)

        # Flags read by the loss / run.py
        self._use_bert_mask = True
        self._use_freq_moe  = False

    def _backbone(self, x, mask=None):
        """raw [B,C,T] (+ optional BERT mask) → (ch_tokens [B,C*Pt,D], reg [B,Pt,D]|None, cls [B,D], Pt)."""
        B, C, T = x.shape
        Pt = T // self.patch_size
        D  = self.embed_dim

        tok = self.patch_embed(x)                              # [B, C*Pt, D] channel-major
        tok = tok + self.pos_embed(Pt, C, x.device).unsqueeze(0)

        if mask is not None:
            mfloat = mask.float().unsqueeze(-1)                # [B, C*Pt, 1]
            mtok   = self.mask_token.expand(B, C * Pt, -1)
            tok    = tok * (1.0 - mfloat) + mtok * mfloat

        # channel-major [B, C*Pt, D] → section grid [B, Pt, C, D]
        grid = tok.reshape(B, C, Pt, D).permute(0, 2, 1, 3).contiguous()
        cls  = self.cls_token.expand(B, -1, -1) if self._use_cls else None

        reg = None
        if self._use_register:
            sec = torch.arange(Pt, device=x.device)
            reg = (self.register_token.expand(B, Pt, -1)
                   + self.register_pos(sec).unsqueeze(0)).unsqueeze(2)   # [B, Pt, 1, D]

        for i in range(self.n_pairs):
            # Frame-wise attention: C channel tokens (+ register) within each section
            if self._use_register:
                fin  = torch.cat([grid, reg], dim=2).reshape(B * Pt, C + 1, D)
                fout = self.frame_blocks[i](fin).reshape(B, Pt, C + 1, D)
                grid = fout[:, :, :C]
                reg  = fout[:, :, C:C + 1]
            else:
                gf   = grid.reshape(B * Pt, C, D)
                grid = self.frame_blocks[i](gf).reshape(B, Pt, C, D)
            # Global attention: CLS + all Pt*C channel tokens (+ Pt register tokens)
            flat = grid.reshape(B, Pt * C, D)
            if self._use_register:
                flat_reg = reg.reshape(B, Pt, D)
                g    = self.global_blocks[i](torch.cat([cls, flat, flat_reg], dim=1))
                cls  = g[:, :1]
                grid = g[:, 1:1 + Pt * C].reshape(B, Pt, C, D)
                reg  = g[:, 1 + Pt * C:].reshape(B, Pt, 1, D)
            elif self._use_cls:
                g    = self.global_blocks[i](torch.cat([cls, flat], dim=1))
                cls  = g[:, :1]
                grid = g[:, 1:].reshape(B, Pt, C, D)
            else:
                grid = self.global_blocks[i](flat).reshape(B, Pt, C, D)

        # back to channel-major [B, C*Pt, D]
        tok = grid.permute(0, 2, 1, 3).reshape(B, C * Pt, D)
        reg_out = self.register_norm(reg.reshape(B, Pt, D)) if self._use_register else None
        tok = self.norm(tok)
        pooled = self.cls_norm(cls.squeeze(1)) if self._use_cls else tok.mean(dim=1)
        return tok, reg_out, pooled, Pt

    def forward(self, x, mask=None, padding_mask=None, return_keep_padding_mask=False):
        tokens, _, cls, _ = self._backbone(x, mask=mask)
        if mask is not None:
            return (tokens, None) if return_keep_padding_mask else tokens
        return cls                                            # [B, D] pooled (CLS) for probing

    def forward_tokens(self, x):
        tokens, _, _, Pt = self._backbone(x, mask=None)
        return tokens, Pt

    def forward_registers(self, x):
        """Per-section pretrained register features [B, Pt, D] (requires use_register)."""
        assert self._use_register, "encoder was not built with use_register: true"
        _, reg, _, Pt = self._backbone(x, mask=None)
        return reg, Pt
