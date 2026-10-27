# FoundDiff MedCNN：八患者＋同步增强，8k

本目录归档实际完成训练的 MedCNN 版本，不是 MLSF-RDDM，也不是重新编写的近似实现。模型、训练、数据加载和冻结指标文件均直接取自本次运行目录；源码 SHA256 见 ARCHIVE_MANIFEST.json。

## 结果

|Step|MedCNN 八患者＋增强 PSNR|八患者 TimeOnly 无增强 PSNR|差值|SSIM|RMSE HU|MAE HU|
|---:|---:|---:|---:|---:|---:|---:|
|2000|40.333292|40.106296|0.226996|0.944119|19.344886|13.097507|
|4000|40.605790|40.684473|-0.078683|0.946227|18.752789|12.650570|
|6000|40.693433|40.781962|-0.088530|0.946937|18.567325|12.496134|
|8000|40.749213|40.841466|-0.092253|0.947394|18.449665|12.547766|

8k 完成，四次固定 L333-Fast64 EMA 验证及最终审计 PASS。最佳 PSNR 为 **40.749213 dB，step 8000**；对应 SSIM 0.947394、RMSE 18.449665 HU、MAE 12.547766 HU。
8k 软组织窗口[-160,240]HU：PSNR 28.440991 dB、SSIM 0.840113、RMSE 15.261581 HU。
训练及验证耗时 107.05 分钟，峰值PyTorch allocated 6.253 GiB；总参数 101,426,285、可训练参数 15,784,301。

相对原两患者无增强 MedCNN@8k，PSNR +0.382015 dB；相对八患者无增强 TimeOnly@8k，PSNR −0.092253 dB。患者数量、增强及模型结构对照并非单变量，不能把差值单独归因为增强或 MedDINO。单seed、单验证患者，不作跨患者显著性结论。

## 模型及数据流

保留 FoundDiff Lite 的 RDDM、原始单卷积RLEB、全部skip和九个TimeOnlyDACB。2.5D输入[I_t,previous LDCT,center LDCT,next LDCT]，单通道中心残差输出。
冻结 MedDINOv3 ViT-B/16，只输入中心LDCT，取[2,5,8,11]层空间patch tokens，四层均为[B,768,32,32]。官方CT均值65.1084213256836、标准差178.01663208007812，保持512分辨率、RoPE FP32。
CNN读取三张LDCT，通道32/32/64/128，对应512/256/128/64尺度；64和128尺度执行局部3×3 MedDINO Cross-Attention，随后门控融合并生成高分辨率结构特征。
九个独立Adapter：E0/E1/D2/D3空间残差注入，E2/E3/MID/D0/D1执行FiLM＋时间门控；保持实际运行位置和零初始化输出投影。代码见 experiments/MedCNN_8patient_Aug_8k/src/medcnn.py 与 src/DADiff.py。

## 训练与增强

训练L067/L096/L109/L143/L192/L286/L291/L310，共4800个中心切片；验证L333固定Fast64原顺序；未访问L506。
seed10、512²、batch2、gradient accumulation2、Adam lr2e-4 betas(.9,.99)、wd0、clip1、FP32、EMA.995逐optimizer step，无scheduler，8000steps，2k/4k/6k/8k验证。
原始 residual MSE、1000 diffusion steps、2-step sampling、sum_scale.01；未改扩散公式，未加入Pyramid等额外loss。
NPY保存原始stored uint16；加载stored−1024 HU、clip[-1000,1000]、dataset[0,1]、模型域[-1,1]。主窗PSNR data_range2000、SSIM win7；软窗data_range400。

当前批次入口是自定义 Stream，而非torch DataLoader。仅训练时：水平翻转p=.5，均匀随机旋转0/90/180/270；前/中/后LDCT与中心NDCT使用完全相同的变换。增强是连续内存中的像素重排，没有裁剪、缩放、插值或额外滤波。验证不增强。
增强使用私有RandomState10，与样本顺序/扩散随机数分开，状态保存在checkpoint。最终审计完整重放32000个中心样本及变换，全部一致。保留原模型内部既有的特征上采样，与图像数据增强区分。

## 关键文件与复现资源

- experiments/MedCNN_8patient_Aug_8k/：实际模型、训练/数据/指标代码、config、固定验证清单、原REPORT及聚合结果。
- paired_preflight.py / finalize_pair.py：实际预检和终检；名字继承原工程，本次只有一个新训练臂。
- vendor/dinov3/：实际使用的MedDINO backbone运行依赖，保留原字节；排除未用分割/扩展源码。授权见vendor/LICENSE_MedDINOv3，来源见MEDDINO_PROVENANCE.json。
- results/COMPARISON.csv：同step历史对照。没有上传逐片图像或大型日志。

真实启动命令：
    cd /home/xujian/research/My-Paper/baselines/FoundDiff_MedCNN_8patient_Aug_8k
    bash run.sh

运行环境：founddiff，Python3.11.16，PyTorch2.8.0+cu128，RTX3090 24GB。
这是实际运行源码归档；代码中的本机路径没有为发布而改写。迁移运行需要自行配置数据、冻结预训练权重、原随机step0与源码比较路径，并重新执行预检，不能直接复用旧PASS文件。
主干只加载原TimeOnly A的未训练step0，新增模块按原seed初始化，预检确认与原MedCNN未训练step0完全一致；没有从任何已训练的TimeOnly/MedCNN继续训练。
MedDINO SHA256：492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2。
最佳EMA仅保留远程：experiments/MedCNN_8patient_Aug_8k/checkpoints/step_8000.pt，key=ema_state_dict。

本归档不包含Mayo数据、CT图像、预训练权重、checkpoint、缓存或大型训练日志；没有修改既有实验。
