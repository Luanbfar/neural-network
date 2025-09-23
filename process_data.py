import argparse
import os

from scripts.preprocess import DataProcessor
from scripts.data_loader import DataLoader


def main(input_csv_path, output_json_path, output_csv_dir):
    """
    Runs the data processing and loading pipeline.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_json_path (str): Path to save the processed JSON data.
        output_csv_dir (str): Directory to save the final CSV datasets.
    """
    # Create the output directories if they don't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    # Instantiate the data processing and loading classes
    data_processor = DataProcessor()
    data_loader = DataLoader()

    # Step 1: Process the CSV data and save it to a JSON file
    print("Starting data processing...")
    data_processor.process_csv(input_csv_path)
    data_processor.save_json(output_json_path)
    print("Data processing complete. Labeled data saved to JSON.")

    # Step 2: Load the JSON data, split it, and export to CSV files
    print("\nStarting data loading and splitting...")
    data_loader.load_data(output_json_path)
    data_loader.export_to_csv(output_csv_dir)
    print("Data loading complete. Datasets saved to CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a data processing pipeline to convert CSV data to labeled JSON and then to split CSV datasets."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
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

    args = parser.parse_args()
    main(args.input_csv, args.output_json, args.output_csv_dir)
