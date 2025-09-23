import json
import random
import csv


class DataLoader:
    """
    A class to load, normalize, and prepare data for machine learning models.

    This class handles the entire data pipeline from a raw JSON file containing
    subject data (age, weight, height) and a corresponding label (CVD probability).
    It normalizes the input features, splits the data into training, testing,
    and validation sets, and can export these datasets to CSV files for
    further analysis or use with other tools.
    """

    def __init__(self):
        """
        Initializes the DataLoader with empty lists for the datasets.

        These lists will be populated after calling the `load_data` method.
        - self.training_data: Stores the training set.
        - self.test_data: Stores the testing set.
        - self.validation_data: Stores the validation set.
        """
        self.training_data = []
        self.test_data = []
        self.validation_data = []

    def normalize(self, age, weight, height) -> list[float]:
        """
        Normalizes input features to a 0-1 range.

        This process scales numerical data to a common range, which is crucial
        for many machine learning algorithms to ensure no single feature
        dominates the learning process. The normalization ranges are hardcoded
        based on typical human physiological values.

        Args:
            age (float): The age of the subject in years.
            weight (float): The weight of the subject in kilograms.
            height (float): The height of the subject in centimeters.

        Returns:
            list: A list containing the normalized values for age, weight, and height.
        """
        # These ranges are based on typical human values
        normalized_age = min(max(age / 100.0, 0), 1)  # 0-100 years
        normalized_weight = min(max(weight / 200.0, 0), 1)  # 0-200 kg
        normalized_height = min(max(height / 250.0, 0), 1)  # 0-250 cm
        return [normalized_age, normalized_weight, normalized_height]

    def load_data(self, filename="data/labeled_data.json"):
        """
        Loads data from a JSON file, normalizes it, and splits it into datasets.

        The function reads a JSON file where data is structured by categories,
        extracts subject information, and normalizes the `age`, `weight`, and
        `height` features. It then shuffles the data and splits it into three
        distinct sets for model training, testing, and validation using a
        70/20/10 ratio.

        Args:
            filename (str): The path to the input JSON file. Defaults to
                            "data/labeled_data.json".

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        try:
            with open(filename) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
            return

        # Collect and normalize all subjects
        all_samples = []
        for subjects in data.values():
            for subject_json in subjects:
                subject = json.loads(subject_json)
                inputs = self.normalize(
                    subject["age"], subject["weight"], subject["height"]
                )
                all_samples.append((inputs, subject["cvd_prob"]))

        random.shuffle(all_samples)

        # Split: 70% train, 20% test, 10% validation
        total = len(all_samples)
        train_end = int(total * 0.7)
        test_end = int(total * 0.9)

        self.training_data = all_samples[:train_end]
        self.test_data = all_samples[train_end:test_end]
        self.validation_data = all_samples[test_end:]

        print(f"Data successfully loaded and split.")
        print(f"Training set size: {len(self.training_data)}")
        print(f"Test set size: {len(self.test_data)}")
        print(f"Validation set size: {len(self.validation_data)}")

    def export_to_csv(self, output_dir="data/"):
        """
        Exports the training, test, and validation datasets to CSV files.

        Each dataset is saved to a separate CSV file within the specified
        output directory. The CSV files include a header row and contain the
        normalized input features (`age_norm`, `weight_norm`, `height_norm`)
        and the target variable (`cvd_prob`).

        Args:
            output_dir (str): The directory where the CSV files will be saved.
                              Defaults to "data".
        """
        datasets = {
            "training": self.training_data,
            "test": self.test_data,
            "validation": self.validation_data,
        }

        for name, data in datasets.items():
            filename = f"{output_dir}/{name}_data.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["age_norm", "weight_norm", "height_norm", "cvd_prob"])

                for inputs, target in data:
                    writer.writerow([inputs[0], inputs[1], inputs[2], target])

        print(f"Exported all datasets to CSV files in '{output_dir}/'")
