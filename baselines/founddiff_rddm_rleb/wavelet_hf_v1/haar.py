"""Fixed one-level orthonormal 2D Haar transform for residual-loss diagnostics."""
import torch
import torch.nn.functional as F

BANDS = ("LL", "LH", "HL", "HH")

class Haar:
    def __init__(self, device, dtype=torch.float32):
        kernels = torch.tensor([
            [[1, 1], [1, 1]],
            [[-1, 1], [-1, 1]],
            [[-1, -1], [1, 1]],
            [[1, -1], [-1, 1]],
        ], dtype=dtype, device=device).unsqueeze(1) * 0.5
        self.kernel = kernels.detach()
        assert self.kernel.shape == (4,1,2,2) and not self.kernel.requires_grad

    def dwt(self, image):
        assert image.ndim == 4 and image.shape[1] == 1
        assert image.shape[-2] % 2 == image.shape[-1] % 2 == 0
        return F.conv2d(image, self.kernel, stride=2, padding=0)

    def idwt(self, bands):
        assert bands.ndim == 4 and bands.shape[1] == 4
        return F.conv_transpose2d(bands, self.kernel, stride=2, padding=0)

    def errors(self, predicted, target):
        p, t = self.dwt(predicted), self.dwt(target)
        parts = {name: F.mse_loss(p[:,i:i+1],t[:,i:i+1])
                 for i,name in enumerate(BANDS)}
        parts["HF"] = (parts["LH"]+parts["HL"]+parts["HH"])/3
        return parts

def unit_test():
    generator = torch.Generator(device="cpu").manual_seed(20260829)
    x = torch.randn((2,1,512,512), generator=generator)
    y = torch.randn((2,1,512,512), generator=generator)
    haar = Haar("cpu")
    recovered = haar.idwt(haar.dwt(x))
    reconstruction_error = (recovered-x).abs().max().item()
    errors = haar.errors(x,y)
    image_mse = F.mse_loss(x,y).item()
    subband_mse = sum(errors[k].item() for k in BANDS)/4
    energy_error = abs(image_mse-subband_mse)
    assert reconstruction_error <= 1e-5
    assert energy_error <= 1e-6
    assert not haar.kernel.requires_grad
    return {"status":"PASS","input_shape":[2,1,512,512],
            "kernel_shape":[4,1,2,2],"kernel_requires_grad":False,
            "max_abs_idwt_dwt_error":reconstruction_error,
            "image_mse":image_mse,"mean_four_subband_mse":subband_mse,
            "mse_parseval_error":energy_error,
            "band_order":list(BANDS)}

