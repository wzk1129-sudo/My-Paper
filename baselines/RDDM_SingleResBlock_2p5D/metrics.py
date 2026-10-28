import numpy as np
from skimage.metrics import structural_similarity
def window_metrics(pred,target,lo,hi):
    p=np.clip(pred,lo,hi).astype(np.float64)
    y=np.clip(target,lo,hi).astype(np.float64)
    mse=np.mean((p-y)**2)
    return {"psnr":float(10*np.log10((hi-lo)**2/mse)),
            "ssim":float(structural_similarity(p,y,win_size=7,data_range=hi-lo,
                         channel_axis=None,gaussian_weights=False,use_sample_covariance=True,K1=0.01,K2=0.03)),
            "rmse_hu":float(np.sqrt(mse)),"mae_hu":float(np.mean(np.abs(p-y)))}
