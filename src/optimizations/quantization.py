"""
Implements Post-Training Dynamic Quantization (INT8) as the framework/
hardware-compatible starting point, plus a
static PTQ path with an explicit calibration step mirroring
Run-Data -> Collect-Stats -> Compute-Scales flow ,
and a QAT scaffold for the accuracy-recovery path when PTQ alone isn't
enough.

"""
import copy

import torch
from torch.quantization import quantize_dynamic


def dynamic_quantize_int8(model: torch.nn.Module, layers_to_quantize=(torch.nn.Linear,)):
    """Fastest PTQ path. No calibration data needed -
    activations quantized on the fly at inference, weights are quantized
    ahead of time. Runs on CPU only.
    """
    model = copy.deepcopy(model).to("cpu").eval()
    quantized = quantize_dynamic(model, set(layers_to_quantize), dtype=torch.qint8)
    return quantized


def static_quantize_int8(model: torch.nn.Module, calibration_dataloader, backend: str = "fbgemm"):
    
    torch.backends.quantized.engine = backend
    model = copy.deepcopy(model).to("cpu").eval()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    prepared = torch.quantization.prepare(model, inplace=False)

    # Calibration: run representative data through so observers record min/max.
    with torch.no_grad():
        for batch in calibration_dataloader:
            inputs = {k: v for k, v in batch.items() if k != "labels" and k != "label"}
            prepared(**inputs)

    quantized = torch.quantization.convert(prepared, inplace=False)
    return quantized


class FakeQuantWrapper(torch.nn.Module):
    """Minimal Quantization-Aware Training simulates
    quantization noise during fine-tuning via fake-quant nodes so the model
    learns weights that are robust to the eventual INT8/INT4 rounding
    """

    def __init__(self, model: torch.nn.Module, backend: str = "fbgemm"):
        super().__init__()
        torch.backends.quantized.engine = backend
        self.model = model
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(self.model, inplace=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def convert(self):
        self.model.eval()
        return torch.quantization.convert(self.model.to("cpu"), inplace=False)


def quantize_bitsandbytes_4bit(checkpoint: str, model_class, **model_kwargs):

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return model_class.from_pretrained(checkpoint, quantization_config=bnb_config, **model_kwargs)
