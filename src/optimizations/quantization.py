"""
Implements Post-Training Dynamic Quantization (INT8) as the framework/
hardware-compatible starting point, plus a

static PTQ path with an explicit calibration step mirroring

Run-Data -> Collect-Stats -> Compute-Scales flow ,

and a QAT scaffold for the accuracy-recovery path when PTQ alone isn't

enough.
"""

# copy is used so that quantization does not modify the original model.
import copy

# PyTorch provides the neural-network modules and quantization utilities.
import torch

# quantize_dynamic performs PyTorch's dynamic post-training quantization.
from torch.quantization import quantize_dynamic


def dynamic_quantize_int8(
    model: torch.nn.Module,
    layers_to_quantize=(torch.nn.Linear,)
):
    """
    Fastest PTQ path.

    PTQ = Post-Training Quantization.
    This means we take an already-trained model and quantize it,
    rather than training the model again.

    No calibration data is needed for dynamic quantization.

    The weights are quantized ahead of time, while activations are
    quantized dynamically during inference.

    This implementation runs on CPU.
    """

    # Make a completely independent copy of the model.
    # This protects the original trained model from being modified.
    #
    # .to("cpu") moves the model to CPU because PyTorch's dynamic
    # quantization path here is intended for CPU inference.
    #
    # .eval() puts the model into evaluation mode because we are
    # preparing it for inference rather than training.
    model = copy.deepcopy(model).to("cpu").eval()

    # Apply dynamic INT8 quantization to the specified layer types.
    #
    # torch.qint8 means the weights are represented using 8-bit
    # signed integers instead of normal FP32 floating-point values.
    #
    # By default, layers_to_quantize contains torch.nn.Linear,
    # which is especially important for Transformer models because
    # they contain many Linear layers.
    quantized = quantize_dynamic(
        model,
        set(layers_to_quantize),
        dtype=torch.qint8
    )

    # Return the new quantized model.
    return quantized


def static_quantize_int8(
    model: torch.nn.Module,
    calibration_dataloader,
    backend: str = "fbgemm"
):
    """
    Static Post-Training Quantization (PTQ).

    Unlike dynamic quantization, static quantization needs
    representative calibration data.

    The calibration data is passed through the model so that
    PyTorch's observers can collect activation statistics such as
    minimum and maximum values.

    Those statistics are then used to determine appropriate
    quantization scales and zero-points.
    """

    # Tell PyTorch which quantization backend to use.
    #
    # FBGEMM is a common backend for x86 CPU quantization.
    torch.backends.quantized.engine = backend

    # Again, work on a copy so the original model is unchanged.
    #
    # Static PTQ in this implementation is performed on CPU.
    #
    # eval() puts the model into inference mode.
    model = copy.deepcopy(model).to("cpu").eval()

    # Create the default quantization configuration for the
    # selected backend.
    #
    # qconfig tells PyTorch how weights and activations should
    # be observed and quantized.
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Prepare the model for calibration.
    #
    # This inserts observer modules into the model.
    # The observers will monitor the values flowing through
    # the model during calibration.
    prepared = torch.quantization.prepare(model, inplace=False)

    # Calibration:
    # Run representative data through the prepared model so
    # the observers can record activation statistics such as
    # minimum and maximum values.
    #
    # torch.no_grad() is used because we are not training here,
    # so gradients are unnecessary.
    with torch.no_grad():

        # Go through each batch of representative calibration data.
        for batch in calibration_dataloader:

            # Remove the labels because they are not model inputs.
            #
            # The batch may contain either "labels" or "label",
            # depending on the dataset/dataloader.
            inputs = {
                k: v
                for k, v in batch.items()
                if k != "labels" and k != "label"
            }

            # Run the inputs through the prepared model.
            #
            # The important thing happening here is not training.
            # The observers inside the prepared model are collecting
            # statistics about the activations.
            prepared(**inputs)

    # Convert the calibrated model into an actual quantized model.
    #
    # PyTorch uses the statistics collected by the observers to
    # determine quantization parameters and replace appropriate
    # floating-point operations/modules with quantized versions.
    quantized = torch.quantization.convert(
        prepared,
        inplace=False
    )

    # Return the final static INT8 model.
    return quantized


class FakeQuantWrapper(torch.nn.Module):
    """
    Minimal Quantization-Aware Training (QAT) wrapper.

    QAT = Quantization-Aware Training.

    Instead of simply quantizing an already-trained model, QAT
    simulates quantization effects during training.

    The model therefore experiences "fake" quantization noise
    while it is being fine-tuned.

    The goal is to let the model adapt to the rounding/precision
    effects that will occur when the final model is converted
    to an actual quantized model.

    This can help recover accuracy when ordinary PTQ causes
    too much accuracy loss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        backend: str = "fbgemm"
    ):

        # Initialize the parent torch.nn.Module.
        super().__init__()

        # Select the quantization backend.
        torch.backends.quantized.engine = backend

        # Store the model inside this wrapper.
        self.model = model

        # Give the model a QAT-specific quantization configuration.
        #
        # Unlike normal PTQ, this configuration is designed to
        # simulate quantization during training.
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(
            backend
        )

        # Insert fake-quantization modules into the model.
        #
        # These modules simulate the effects of quantization while
        # the model is still using floating-point tensors.
        #
        # The model can then be fine-tuned so it becomes more robust
        # to the eventual INT8 quantization.
        torch.quantization.prepare_qat(
            self.model,
            inplace=True
        )

    def forward(self, *args, **kwargs):

        # Pass any inputs directly to the wrapped model.
        #
        # *args allows positional arguments and **kwargs allows
        # keyword arguments such as input_ids, attention_mask, etc.
        return self.model(*args, **kwargs)

    def convert(self):

        # Switch the model to evaluation mode because we are
        # preparing the final model for inference.
        self.model.eval()

        # Convert the QAT-prepared model into an actual quantized
        # model.
        #
        # The final conversion happens on CPU here.
        return torch.quantization.convert(
            self.model.to("cpu"),
            inplace=False
        )


def quantize_bitsandbytes_4bit(
    checkpoint: str,
    model_class,
    **model_kwargs
):
    """
    Load a Hugging Face model using BitsAndBytes 4-bit quantization.

    This is different from the INT8 PyTorch paths above.

    Instead of converting an already-loaded model using PyTorch's
    quantization utilities, BitsAndBytes allows a Hugging Face
    model to be loaded directly using 4-bit weights.

    NF4 (NormalFloat4) is a 4-bit quantization format designed
    for neural-network weights.

    This function is mainly a low-memory model-loading path.
    """

    # Import Hugging Face's BitsAndBytes configuration class.
    from transformers import BitsAndBytesConfig

    # Create the 4-bit quantization configuration.
    bnb_config = BitsAndBytesConfig(

        # Load the model weights using 4-bit representation.
        load_in_4bit=True,

        # Use NF4, a 4-bit data type designed for quantizing
        # normally distributed neural-network weights.
        bnb_4bit_quant_type="nf4",

        # Perform computations using FP16 while the stored
        # weights remain quantized to 4-bit.
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the pretrained model using the 4-bit configuration.
    #
    # checkpoint = the Hugging Face model/checkpoint name or path.
    #
    # model_class = the class used to load that model.
    #
    # **model_kwargs passes any additional model-loading arguments.
    return model_class.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        **model_kwargs
    )