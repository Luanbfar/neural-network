import argparse
import os
import sys  # NEW: Import sys for argument handling and exit control

from scripts.preprocess import DataProcessor
from scripts.data_loader import DataLoader


def main(input_csv_path, output_json_path, output_csv_dir):
    """
    Runs the data processing and loading pipeline.
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    data_processor = DataProcessor()
    data_loader = DataLoader()

    print("Starting data processing...")
    data_processor.process_csv(input_csv_path)
    data_processor.save_json(output_json_path)
    print("Data processing complete. Labeled data saved to JSON.")

    print("\nStarting data loading and splitting...")
    data_loader.load_data(output_json_path)
    data_loader.export_to_csv(output_csv_dir)
    print("Data loading complete. Datasets saved to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the data processing pipeline or normalize a single data point."
    )
    # --- Main pipeline arguments ---
    parser.add_argument(
        "--input_csv",
        type=str,
        # NEW: required=False since it can now be optional
        required=False,
        help="Path to the input CSV file (e.g., 'data/NHANES-2017-2018-height-weight.csv').",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="data/labeled_data.json",
        help="Path to save the labeled data as a JSON file.",
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default="data",
        help="Directory to save the final training, test, and validation CSV files.",
    )

    # NEW: Argument for normalization function
    parser.add_argument(
        "--normalize",
        nargs=3,  # Expects exactly 3 values
        type=float,  # Converts values to float
        metavar=('AGE', 'WEIGHT_KG', 'HEIGHT_CM'),
        help="Normalize a single data point. Provide age, weight (kg), and height (cm)."
    )

    args = parser.parse_args()

    # NEW: Conditional logic to decide which action to run
    if args.normalize:
        # If --normalize was passed, run this section
        if len(args.normalize) != 3:
            print("Error: The --normalize argument requires exactly 3 values: age, weight, and height.", file=sys.stderr)
            sys.exit(1)
        
        # Assuming normalize is a method of DataLoader
        loader = DataLoader()
        
        # Unpack the argument list into the function
        normalized_data = loader.normalize(*args.normalize)
        
        # Print the result in a format easy to parse by C++
        print(f"{normalized_data[0]},{normalized_data[1]},{normalized_data[2]}")
        sys.exit(0)  # End the script successfully

    # If --normalize was not passed, run the main pipeline
    # Validation to ensure --input_csv is provided for the pipeline
    if not args.input_csv:
        parser.error("--input_csv is required when not using --normalize.")
    
    main(args.input_csv, args.output_json, args.output_csv_dir)
