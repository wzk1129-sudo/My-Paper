"""Deterministic synthetic tests; no patient data or checkpoint writes."""
import torch
from local_detail_encoder import LocalDetailEncoder
from local_cross_attention import LocalMedCrossAttention
from mlsf_model import CORRECTED


def main():
    torch.set_num_threads(2)
    x = torch.rand(1, 1, 512, 512)
    encoder = LocalDetailEncoder()
    l256, l128 = encoder(x, x, x)
    assert l256.shape == (1, 32, 256, 256)
    assert l128.shape == (1, 64, 128, 128)
    attention = LocalMedCrossAttention()
    index = attention.neighbor_index
    assert index.shape == (16384, 9)
    assert index.min() >= 0 and index.max() <= 1023
    assert index[0].tolist() == [0, 0, 1, 0, 0, 1, 32, 32, 33]
    assert index[-1].tolist() == [990, 991, 991, 1022, 1023, 1023, 1022, 1023, 1023]
    output = attention(l128, torch.randn(1, 64, 32, 32))
    assert output.shape == l128.shape and torch.isfinite(output).all()
    assert set(CORRECTED) == {"E1", "E2", "D1", "D2"}
    print("TEST_MLSF_PASS")


if __name__ == "__main__":
    main()
