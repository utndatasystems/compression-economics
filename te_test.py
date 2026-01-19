import torch
import transformer_engine.pytorch as te

print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 名稱:", torch.cuda.get_device_name(0))

# 建立 Linear Layer
layer = te.Linear(in_features=1024, out_features=1024, bias=True, params_dtype=torch.float16).cuda()

# 輸入 FP16 tensor
x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)

try:
    with te.fp8_autocast(enabled=True):  # 在 FP8 模式中 forward
        y = layer(x)
    print("FP8 測試成功，輸出形狀:", y.shape)
except Exception as e:
    print("FP8 測試失敗:", e)
