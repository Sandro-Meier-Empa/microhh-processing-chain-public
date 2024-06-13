#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.append(os.path.join("/users", os.environ["USER"], "processing_chain", "src"))
import processing as proc
import initialisation as init


def process_case(case_name):
    case_name, case_suffix = case_name.split(".", 1)
    case = init.Case(
        case_name=case_name,
        suffix=case_suffix,
        scratch_path="/scratch/snx3000",
        store_path="/store/empa/em05/smeier/microhh_output",
        location="scratch",
    )

    post_processing = proc.PostProcessing(case)

    try:
        post_processing.run_postprocessing(gas="ch4", height=10)
    except Exception as err:
        print(f"Error processing {case_suffix}: {err}")


def main(cases):
    """Run 'process_case' for all cases in parallel"""
    with ThreadPoolExecutor() as executor:
        future_to_case = {
            executor.submit(process_case, case_name): case_name for case_name in cases
        }
        for future in as_completed(future_to_case):
            case_suffix = future_to_case[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{case_suffix} generated an exception: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cases in parallel.")
    parser.add_argument(
        "--case-names",
        required=True,
        nargs="+",
        help="List of case names (case_name.case_suffix)",
    )

    args = parser.parse_args()

    main(args.case_names)
