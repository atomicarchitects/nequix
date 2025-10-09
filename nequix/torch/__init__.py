import e3nn
import torch

# Flags for torch.compile
e3nn.set_optimization_defaults(jit_script_fx=False)

# Set default precision
torch.set_float32_matmul_precision("highest")

# Flags for torch.compile
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
torch.fx.experimental._config.use_duck_shape = False
