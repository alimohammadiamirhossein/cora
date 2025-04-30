import argparse


def add_general_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.set_defaults(fp16=False)
    parser.add_argument("--fp16", action="store_true")
    # parser.add_argument("--seeds", type=int, nargs='+', default=[7], help="List of seed values (e.g., --seed 22 42)")
    parser.add_argument("--seed", type=int, default=7, help="Seed value for random number generation.")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--eval_dataset_folder", type=str, default="dataset")
    parser.add_argument("--num_of_timesteps", type=int, default=5)  # 3 or 4


def add_extra_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale value.")
    parser.add_argument("--apply_dift_correction", action="store_true", help="Apply DIFT correction.")
    parser.set_defaults(apply_dift_correction=False)
    parser.add_argument("--w1", type=float, default=1.9, help="Weight for CTRL-X mode.")
    parser.add_argument("--support_new_object", action="store_true", help="Enable support for new object detection.")
    parser.add_argument("--mode", type=str, default="slerp_dift",
                        help="Attention Type (e.g., normal, slerp, lerp, ...).")
    parser.add_argument("--dift_timestep", type=int, default=400, help="DIFT timestep.")
    parser.add_argument("--movement_intensifier", type=float, default=0.2, help="Movement intensifier factor.")
    parser.add_argument("--structural_alignment", action="store_true", help="Enable structural alignment.")


def add_editing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_norm_zs", type=float, nargs="+", default=[-1, -1, -1, 15.5],
                        help="A list of floats for max_norm_zs.")
    parser.add_argument("--noise_shift_delta", type=float, default=1)
    parser.add_argument("--noise_timesteps", type=int, nargs="+", default=[799, 499, 199, 0],
                        help="A list of ints for noise_timesteps.")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[999, 799, 499, 199],
                        help="A list of ints for timesteps.")
    parser.add_argument("--num_steps_inversion", type=int, default=5)
    parser.add_argument("--step_start", type=int, default=1)


def check_args(args):
    if args.num_of_timesteps not in [3, 4, 5, 10]:
        raise ValueError("num_timesteps must be 3, 4, or 5 or 10")

    if args.timesteps is not None:
        num_steps_actual = len(args.timesteps)
    else:
        num_steps_actual = args.num_steps_inversion - args.step_start

    if isinstance(args.max_norm_zs, (int, float)):
        args.max_norm_zs = [args.max_norm_zs] * num_steps_actual

    assert (
            len(args.max_norm_zs) == num_steps_actual
    ), f"len(args.max_norm_zs) ({len(args.max_norm_zs)}) != num_steps_actual ({num_steps_actual})"

    assert args.noise_timesteps is None or len(args.noise_timesteps) == (
        num_steps_actual
    ), f"len(args.noise_timesteps) ({len(args.noise_timesteps)}) != num_steps_actual ({num_steps_actual})"


def get_args():
    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    add_editing_arguments(parser)
    add_extra_arguments(parser)
    args = parser.parse_args()
    check_args(args)
    return args

