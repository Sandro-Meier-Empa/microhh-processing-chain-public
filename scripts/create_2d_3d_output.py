import argparse
import logging
import microhh_processing_chain.initialisation as init
import microhh_processing_chain.postprocessing as post

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MicroHH simulation output and generate 2D/3D datasets."
    )

    parser.add_argument(
        "--file_directory",
        type=str,
        required=True,
        help="Path to the MicroHH case directory...",
    )

    parser.add_argument(
        "--sample_time",
        type=int,
        help="Sampling time interval for 3D output.",
    )

    parser.add_argument(
        "--vars_3d",
        nargs="+",
        default=["u", "v", "w"],
        help="Variables to include in 3D output (default: u v w).",
    )

    parser.add_argument(
        "--vars_2d",
        nargs="+",
        default=["ch4_path"],
        help="Variables to include in 2D output (default: ch4_path).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.info(f"Processing case: {args.directory}")
    case = init.read_case(args.directory)
    post_processing = post.PostProcessing(case)
    post_processing.create_3d_output(vars=args.vars_3d, sample_time=args.sample_time)
    post_processing.create_2d_output(vars=args.vars_2d)

    logging.info(f"Finished case in {args.directory}")
