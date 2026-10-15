"""
src/models/__init__.py
"""
import inspect
from .autoencoder import ConvAutoencoder
from .padim import PaDiM
from .patchcore import PatchCore
from .student_teacher import StudentTeacher

MODEL_REGISTRY = {
    "autoencoder": ConvAutoencoder,
    "padim": PaDiM,
    "patchcore": PatchCore,
    "student_teacher": StudentTeacher,
}


def build_model(name: str, **kwargs):
    """Instantiate a model by name, filtering kwargs supported by the class."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
    model = cls(**valid_args)

    # If device was passed and model is a torch.nn.Module, move it to device
    if "device" in kwargs and hasattr(model, "to"):
        import torch
        device_str = kwargs["device"]
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    return model
