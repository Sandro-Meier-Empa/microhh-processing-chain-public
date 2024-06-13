#!/usr/bin/env python3

import os
import sys
import argparse

sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
import processing as proc
import initialisation as init


def process_case(case_name, case_suffix):
    case = init.Case(
        case_name=case_name,
        suffix=case_suffix,
        scratch_path="/scratch/snx3000",
        store_path="/store/empa/em05/smeier/microhh_output",
    )

    post_processing = proc.PostProcessing(case)

    try:
        post_processing.run_postprocessing(gas="ch4", height=10)
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
