from argparse import ArgumentParser
from pathlib import Path

from nequix.model import load_model as load_model_jax
from nequix.model import save_model as save_model_jax
from nequix.torch_impl.model import load_model as load_model_torch
from nequix.torch_impl.model import save_model as save_model_torch
from nequix.torch_impl.utils import convert_model_jax_to_torch
from nequix.torch_impl.utils import convert_model_torch_to_jax


def main():
    parser = ArgumentParser()
    parser.add_argument("input_path", type=Path, help="Path to input model")
    parser.add_argument("output_path", type=Path, help="Path to output model")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    input_backend = "jax" if input_path.suffix == ".nqx" else "torch"
    output_path = Path(args.output_path)
    output_backend = "torch" if output_path.suffix == ".pt" else "jax"

    if input_backend == "jax" and output_backend == "torch":
        model, config = load_model_jax(input_path)
        # NB: use_kernel doesn't matter since we're just saving
        torch_model, torch_config = convert_model_jax_to_torch(model, config, use_kernel=False)
        save_model_torch(output_path, torch_model, torch_config)
    elif input_backend == "torch" and output_backend == "jax":
        torch_model, torch_config = load_model_torch(input_path)
        jax_model, jax_config = convert_model_torch_to_jax(
            torch_model, torch_config, use_kernel=False
        )
        save_model_jax(output_path, jax_model, jax_config)
    else:
        raise ValueError(f"invalid input and output backends: {input_backend} and {output_backend}")
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
