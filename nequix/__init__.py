import jax

# NB: this is required for compatibility across different generations of GPUs.
# Ampere and later (e.g. A100, H100) generations use tf32 for matmuls as the
# default, where older generations such as Volta (e.g. V100) use fp32. Using the
# "highest" precision will force the use of fp32 for matmuls, ensuring
# compatibility across different GPUs.  However, this will slow down training
# and inference speed on newer GPUs.  For more information, see:
# https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision
jax.config.update("jax_default_matmul_precision", "highest")
