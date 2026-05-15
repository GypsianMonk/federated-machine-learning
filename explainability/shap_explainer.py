import shap
import torch

def explain(model, sample):
    model.eval()
    
    sample_tensor = torch.tensor(sample, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, sample_tensor[:10])
    shap_values = explainer.shap_values(sample_tensor[:1])

    return shap_values
