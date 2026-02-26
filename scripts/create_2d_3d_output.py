import sys
import logging
import microhh_processing.initialisation as init
import microhh_processing.postprocessing as post

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("No case path provided.")
        sys.exit(1)

    path = sys.argv[1]

    logging.info(f"Processing case: {path}")
    case = init.read_case(path)
    post_processing = post.PostProcessing(case)
    post_processing.create_2d_output(vars=["ch4_path"])
    post_processing.create_3d_output(vars=["u", "v", "w"])

    logging.info(f"Finished case: {path}")
