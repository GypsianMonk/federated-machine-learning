import torch
from model.cnn_lstm import CNNLSTM
from data.synthetic_ecg import generate_ecg_data
import shap

X, y = generate_ecg_data(samples=20)
model = CNNLSTM()
model.eval()

try:
    print("Testing DeepExplainer")
    explainer = shap.DeepExplainer(model, torch.tensor(X[:10], dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X[:1], dtype=torch.float32))
    print("DeepExplainer works!")
except Exception as e:
    print("DeepExplainer failed:", e)

try:
    print("Testing GradientExplainer")
    explainer = shap.GradientExplainer(model, torch.tensor(X[:10], dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X[:1], dtype=torch.float32))
    print("GradientExplainer works!")
except Exception as e:
    print("GradientExplainer failed:", e)

try:
    print("Testing KernelExplainer Flattened")
    original_shape = X.shape[1:]
    def f(X_2d):
        X_3d = X_2d.reshape(-1, *original_shape)
        X_tensor = torch.tensor(X_3d, dtype=torch.float32)
        return model(X_tensor).detach().numpy()
    
    sample_2d = X.reshape(X.shape[0], -1)
    explainer = shap.KernelExplainer(f, sample_2d[:10])
    shap_values = explainer.shap_values(sample_2d[:1])
    print("KernelExplainer Flattened works!")
except Exception as e:
    print("KernelExplainer Flattened failed:", e)
