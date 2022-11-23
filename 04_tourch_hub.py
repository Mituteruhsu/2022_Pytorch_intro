# torch 的 Hub 有許多 model tool, 可以使用多個已建置好的模型帶入 
import torch
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
# model.eval()
print(model.eval())