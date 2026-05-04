#!/usr/bin/env python3
import argparse

from microhh_processing_chain import initialisation as init
from microhh_processing_chain import postprocessing as post


def process_case(case_name, case_suffix):
    case = init.Case(
        case_name=case_name,
        suffix=case_suffix,
        scratch_path="/scratch/snx3000",
        store_path="/capstor/store/cscs/empa/em05/smeier/microhh_output",
        location="scratch",
    )

    post_processing = post.PostProcessing(case)

    try:
        post_processing.zip_binary_files(remove_files=True)
    except Exception as err:
        print(f"Error processing {case_suffix}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Process some cases.")
    parser.add_argument(
        "--case-name", required=True, help="Name of the case to process"
    )
    parser.add_argument(
        "--case-suffix", required=True, help="Suffix of the case to process"
    )

    args = parser.parse_args()

    process_case(args.case_name, args.case_suffix)


if __name__ == "__main__":
    main()
