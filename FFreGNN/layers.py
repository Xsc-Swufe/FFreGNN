
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
import torch.nn as nn
np.set_printoptions(threshold=np.inf)
from training.tools import *
import math
import torch

















class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer to capture temporal order.
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [N, T, F'].
        Returns:
            torch.Tensor: Input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln_real = nn.LayerNorm(normalized_shape)
        self.ln_imag = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        real = x.real
        imag = x.imag

        real_norm = self.ln_real(real)
        imag_norm = self.ln_imag(imag)

        x_norm = torch.complex(real_norm, imag_norm)
        return x_norm














class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.Linear_real = torch.nn.Linear(in_features, out_features, bias=bias)
        self.Linear_img = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        real_real = self.Linear_real(input.real)
        img_real = self.Linear_img(input.real)
        real_img = self.Linear_real(input.imag)
        img_img = self.Linear_img(input.imag)
        return real_real - img_img + 1j * (real_img + img_real)

class TemporalFeatureExtractor(nn.Module):
    """
    Extracts deep temporal features from reconstructed time series using a Transformer-based
    architecture.

    Args:
        T (int): Time steps of the original time series.
        feature_dim (int): Dimension of the output feature vector (F'). Must be multiple of 4.
    """
    def __init__(self, T, feature_dim):
        super(TemporalFeatureExtractor, self).__init__()

        if feature_dim % 4 != 0:
            raise ValueError("feature_dim must be a multiple of 4 for the Transformer module.")

        self.T = T
        self.feature_dim = feature_dim
        in_channels = 1  # Reconstructed signal has 1 channel

        # Input projection to Transformer dimension
        self.input_proj = nn.Conv1d(in_channels, feature_dim, kernel_size=1)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_dim, T)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,  # Assuming feature_dim is divisible by 4
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Residual connection
        self.residual_proj = nn.Conv1d(in_channels, feature_dim, kernel_size=1)

        # Layer normalization
        self.layer_norm = nn.LayerNorm([feature_dim, T], eps=1e-4)

        # Final linear layer to ensure output shape
        self.linear = nn.Linear(T, 1, bias=True)

    def forward(self, x):
        """
        Extracts deep temporal features from reconstructed signal using Transformer.

        Args:
            x (torch.Tensor): Reconstructed time series, shape [N, T].

        Returns:
            torch.Tensor: Deep temporal feature vector, shape [N, F'].
        """
        # Reshape for 1D convolution: [N, T] -> [N, 1, T]
        N, T = x.shape
        x = x.unsqueeze(1)  # [N, 1, T]

        # Project input to feature_dim
        x_transformed = self.input_proj(x)  # [N, F', T]

        # Transpose for Transformer: [N, F', T] -> [N, T, F']
        x_transformed = x_transformed.permute(0, 2, 1)

        # Add positional encoding
        x_transformed = self.pos_encoder(x_transformed)

        # Transformer encoder
        transformer_out = self.transformer_encoder(x_transformed)  # [N, T, F']

        # Transpose back: [N, T, F'] -> [N, F', T]
        transformer_out = transformer_out.permute(0, 2, 1)

        # Residual connection
        residual = self.residual_proj(x)  # [N, F', T]

        # Combine and normalize
        h = F.relu(transformer_out + residual)
        h = self.layer_norm(h)  # [N, F', T]

        # Linear layer to reduce temporal dimension
        h_final = self.linear(h.reshape(-1, T))  # [N*F', T] -> [N*F', 1]
        h_final = h_final.reshape(N, -1)  # [N, F']

        return h_final



class KnowledgeGuidedFrequencyAlignment(nn.Module):
    """
    Knowledge-guided Frequency Paradigm Alignment Module

    Transforms multi-dimensional time-series into frequency domain, aligns spectra
    using a complex-valued knowledge base with attention mechanism, and extracts
    temporal features via Transformer.
    """
    def __init__(self, T: int, C: int, P: int, feature_dim: int, bias=True):
        super().__init__()
        self.T = int(T)
        self.C = int(C)
        self.P = int(P)  # Number of knowledge prototypes
        self.L = self.T // 2 + 1  # Effective frequency length
        self.embed_dim = self.L  # Embedding dimension for complex-valued MLPs
        self.feature_dim = feature_dim  # Dimension for Transformer output

        # Learnable frequency-domain knowledge base M_Θ ∈ ℂ^(P × L)
        self.knowledge_base = nn.Parameter(torch.randn(self.P, self.L, dtype=torch.cfloat))

        # Complex-valued MLPs for query, key, and value projections
        self.phi_q = ComplexLinear(self.C * self.L, self.embed_dim, bias=bias)
        self.phi_k = ComplexLinear(self.L, self.embed_dim, bias=bias)
        self.phi_v = ComplexLinear(self.L, self.embed_dim, bias=bias)  # Ensure output dim is L

        # Scaling factor for attention
        self.scaling = self.embed_dim ** -0.5

        # Transformer-based temporal feature extractor
        self.temporal_extractor = TemporalFeatureExtractor(T, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [T, N, C] real-valued time series.

        Returns:
            orig_amplitude: [N, C, L] original amplitude spectrum.
            orig_phase: [N, C, L] original phase spectrum.
            refined_amplitude: [N, L] refined amplitude spectrum.
            refined_phase: [N, L] refined phase spectrum.
            temporal_features: [N, F'] deep temporal features.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [T, N, C], got {tuple(x.shape)}")
        T_in, N, C_in = x.shape
        if T_in != self.T:
            raise ValueError(f"Input T={T_in} != {self.T}")
        if C_in != self.C:
            raise ValueError(f"Input C={C_in} != {self.C}")

        # Align dtype/device
        dtype = self.phi_q.Linear_real.weight.dtype
        device = self.phi_q.Linear_real.weight.device
        x = x.to(device=device, dtype=dtype)

        # Step 1: Transform to frequency domain via FFT
        # [T, N, C] -> [N, C, T] for FFT along time dimension
        x_nct = x.permute(1, 2, 0).contiguous()  # [N, C, T]
        h_nc = torch.fft.fft(x_nct, dim=-1)  # [N, C, T] -> [N, C, T] complex
        h_nc = h_nc[:, :, :self.L]  # Take non-redundant portion [N, C, L]

        # Step 2: Extract original amplitude and phase
        orig_amplitude = torch.abs(h_nc)  # [N, C, L]
        orig_phase = torch.angle(h_nc)  # [N, C, L]

        # Step 3: Flatten and concatenate channel spectra
        h_n = h_nc.reshape(N, -1)  # [N, C*L]

        # Step 4: Generate query, key, and value vectors
        q_n = self.phi_q(h_n)  # [N, L]
        k_p = self.phi_k(self.knowledge_base)  # [P, L]
        v_p = self.phi_v(self.knowledge_base)  # [P, L]

        # Step 5: Compute complex-valued similarity scores
        attn_weights = torch.matmul(q_n, torch.conj(k_p).transpose(-2, -1)) * self.scaling  # [N, P]
        real_scores = torch.real(attn_weights)  # [N, P]
        alpha_np = F.softmax(real_scores, dim=-1)  # [N, P]

        # Step 6: Reconstruct aligned spectral representation
        h_tilde_n = torch.matmul(alpha_np.type(torch.complex64), v_p)  # [N, L]

        # Step 7: Extract refined amplitude and phase
        refined_amplitude = torch.abs(h_tilde_n)  # [N, L]
        refined_phase = torch.angle(h_tilde_n)  # [N, L]

        # Step 8: Inverse FFT to reconstruct time-domain signal
        out = torch.fft.ifft(h_tilde_n, n=self.T, dim=-1).real  # [N, T], take real part

        # Step 9: Extract deep temporal features using Transformer
        temporal_features = self.ln(self.temporal_extractor(out))  # [N, F']

        return orig_amplitude, orig_phase, refined_amplitude, refined_phase, temporal_features

















class DynamicRelationExtraction(nn.Module):
    """
    Dynamic Relation Extraction based on spectral coherence theory.

    Args:
        C (int): Number of dominant frequency components.
        tau_factor (float): Factor for dynamic thresholding (default: 0.5).
    """

    def __init__(self, C, tau_factor=1):
        super(DynamicRelationExtraction, self).__init__()
        self.C = C
        self.tau_factor = tau_factor

    def forward(self, P_ref, A_ref):
        """
        Extract dynamic relations using spectral coherence.

        Args:
            P_ref (torch.Tensor): Refined phase spectra, shape [N, K].
            A_ref (torch.Tensor): Refined amplitude spectra, shape [N, K].

        Returns:
            torch.Tensor: Sparsified adjacency matrix, shape [N, N].
        """
        N, K = A_ref.shape

        # Step 1: Parallel generation of dominant frequency masks
        _, indices = torch.topk(A_ref, k=self.C, dim=1)  # [N, C]
        masks = torch.zeros(N, K, device=A_ref.device, dtype=torch.bool)
        masks.scatter_(1, indices, True)  # [N, K]

        # Step 2: Apply masks to filter non-dominant components
        A_masked = A_ref * masks.float()  # [N, K]
        P_masked = P_ref * masks.float()  # [N, K]

        # Step 3: Compute raw relation matrix via broadcasted tensor operations
        A_n = A_masked.unsqueeze(1)  # [N, 1, K]
        A_m = A_masked.unsqueeze(0)  # [1, N, K]
        P_diff = P_masked.unsqueeze(1) - P_masked.unsqueeze(0)  # [N, N, K]
        num = torch.sum(A_n * A_m * torch.cos(P_diff), dim=2)  # [N, N]

        A_n_sq = torch.sum(A_masked ** 2, dim=1, keepdim=True)  # [N, 1]
        A_m_sq = torch.sum(A_masked ** 2, dim=1, keepdim=True).transpose(0, 1)  # [1, N]
        denom = torch.sqrt(A_n_sq * A_m_sq + 1e-10)  # [N, N]

        R = num / denom  # [N, N]

        # Step 4: Normalization and dynamic thresholding for sparsification
        row_sums = torch.sum(R, dim=1, keepdim=True) + 1e-10
        R_prime = R / row_sums  # [N, N]

        mean_positive = torch.mean(R_prime[R_prime > 0])
        tau = self.tau_factor * mean_positive if mean_positive > 0 else self.tau_factor
        R_hat = torch.where(R_prime >= tau, R_prime, torch.zeros_like(R_prime))

        return R_hat












class StateAwareGraphRouting(nn.Module):
    """
    State-Aware Graph Routing Network for dynamic risk propagation in financial networks.

    Args:
        input_dim (int): Dimension of input node features (F').
        num_modes (int): Number of propagation modes (G).
        d_k (int): Dimension of key vectors for attention scaling.
        tau (float): Temperature parameter for Gumbel-Softmax.
    """

    def __init__(self, input_dim, num_modes, d_k, tau=1.0):
        super(StateAwareGraphRouting, self).__init__()
        self.input_dim = input_dim
        self.num_modes = num_modes
        self.d_k = d_k
        self.tau = tau

        # Linear layers for individual and systemic feature extraction
        self.W_ind = nn.Linear(d_k, d_k)
        self.W_gate = nn.Linear(d_k, d_k)
        self.W_sys = nn.Linear(2, d_k)

        # Propagation mode manifold
        self.W_s = nn.Parameter(torch.randn(num_modes, d_k, d_k))
        self.W_k = nn.Linear(d_k * d_k, d_k)

    def forward(self, z_n, P_ref, P_in, A_ref, A_in, R_nm):
        """
        Forward pass of the state-aware graph routing network.

        Args:
            z_n (torch.Tensor): Individual temporal state (all node states), shape [N, F'].
            P_ref (torch.Tensor): Refined phase spectrum, shape [N, K].
            P_in (torch.Tensor): Input phase spectrum, shape [N, K].
            A_ref (torch.Tensor): Refined amplitude spectrum, shape [N, K].
            A_in (torch.Tensor): Input amplitude spectrum, shape [N, K].
            R_nm (torch.Tensor): Precomputed relation strength matrix, shape [N, N].

        Returns:
            torch.Tensor: Updated node state, shape [N, F'].
        """
        N, K = A_ref.size()




        P_ref = P_ref.unsqueeze(1)  # [198, 1, 16]
        A_ref = A_ref.unsqueeze(1)  # [198, 1, 16]

        delta_P_n = (P_in - P_ref).pow(2).mean(dim=[1, 2]).unsqueeze(1)  # 按节点求平均 [198]
        delta_A_n = (A_in - A_ref).pow(2).mean(dim=[1, 2]).unsqueeze(1)  # [198]

        # Contextualized State Query (Equation 3)
        gate = torch.sigmoid(self.W_gate(z_n))
        sys_feature = self.W_sys(torch.cat([delta_A_n, delta_P_n], dim=1))
        q_n = self.W_ind(z_n) + gate * sys_feature  # [N, F']

        # Mode Key Vectors (Equation 4)
        W_s_flat = self.W_s.view(self.num_modes, -1)
        k_g = self.W_k(W_s_flat)  # [G, d_k]

        # Attention scores (Equation 5)
        q_n_expanded = q_n.unsqueeze(1)  # [N, 1, d_k]
        k_g_expanded = k_g.unsqueeze(0)  # [1, G, d_k]
        s_ng = torch.matmul(q_n_expanded, k_g_expanded.transpose(1, 2)) / (self.d_k ** 0.5)  # [N, 1, G]
        s_ng = s_ng.squeeze(1)  # [N, G]

        # Gumbel-Softmax for routing weights (Equation 6)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(s_ng) + 1e-10) + 1e-10)
        alpha_ng = F.softmax((s_ng + gumbel_noise) / self.tau, dim=-1)  # [N, G]

        # State-Aware Propagation (Equation 7)
        W_s_weighted = torch.einsum('ng, gij -> nij', alpha_ng, self.W_s)  # [N, F', F']
        propagated = torch.bmm(W_s_weighted, z_n.unsqueeze(-1)).squeeze(-1)
        h_n = F.leaky_relu(torch.matmul(R_nm, propagated))

        return h_n, s_ng




