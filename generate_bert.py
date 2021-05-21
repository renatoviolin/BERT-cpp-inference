# %% ---------------------------------------------
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.onnx

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased", torchscript=True)
model.eval()

dummy_tensor = torch.randint(0, 30522, (1, 512))


# %% ---------------------------------------------
with torch.no_grad():
    traced_model = torch.jit.trace(model, dummy_tensor)
    torch.jit.save(traced_model, "model/large_lm.pt")


# %% ---------------------------------------------
batch_size = 1
torch_out = model(dummy_tensor)
torch.onnx.export(model,               # model being run
                  dummy_tensor,                         # model input (or a tuple for multiple inputs)
                  "bert_onnx.pt",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

# %%
