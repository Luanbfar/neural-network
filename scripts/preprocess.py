import csv
import json
import math


class DataProcessor:
    """
    A class to process raw health data from a CSV, calculate key metrics,
    and categorize subjects based on BMI.

    This class handles the end-to-end data pipeline from a raw CSV containing
    height and weight measurements to a structured JSON output with calculated
    BMI and cardiovascular disease (CVD) risk probabilities.
    """

    def __init__(self):
        """
        Initializes the DataProcessor with a dictionary to hold categorized subject data.

        The dictionary 'self.data' is structured with keys representing
        standard BMI categories, and values as lists that will store processed
        subject information.
        """
        self.data = {
            "underweight": [],
            "normal": [],
            "overweight": [],
            "obese": [],
            "morbid_obese": [],
        }

    def calculate_bmi(self, weight, height) -> float:
        """
        Calculates the Body Mass Index (BMI) from weight and height.

        Args:
            weight (float): Subject's weight in kilograms (kg).
            height (float): Subject's height in centimeters (cm).

        Returns:
            float: The calculated BMI, rounded to two decimal places.
        """
        height_m = height / 100
        return round(weight / (height_m * height_m), 2)

    def calculate_cvd_risk(self, bmi, age) -> float:
        """
        Calculates the probability of cardiovascular disease (CVD) risk.

        The risk is determined using a model that combines BMI and age, based
        on a logistic function for age and a quadratic function for BMI.

        Args:
            bmi (float): Subject's Body Mass Index.
            age (float): Subject's age in years.

        Returns:
            float: The calculated CVD risk probability, a value between 0.0 and 1.0.
        """
        bmi_risk = 0.0023 * bmi**2 - 0.0797 * bmi + 1.6927
        age_risk = 0.8861 / (1 + math.exp(-0.1164 * (age - 52.8598)))
        return min(round(bmi_risk * age_risk, 4), 1.0)

    def categorize_bmi(self, bmi) -> str:
        """
        Categorizes a BMI value into a health-based category.

        The categories follow standard health guidelines.

        Args:
            bmi (float): Subject's Body Mass Index.

        Returns:
            str: The BMI category. Possible values are "underweight", "normal",
                 "overweight", "obese", or "morbid_obese".
        """
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25.0:
            return "normal"
        elif bmi < 30.0:
            return "overweight"
        elif bmi < 40.0:
            return "obese"
        else:
            return "morbid_obese"

    def process_csv(self, filename="data/NHANES-2017-2018-height-weight.csv"):
        """
        Processes a CSV file, calculating BMI and CVD risk for each subject.

        Reads a CSV file with columns 'id', 'age', 'weight', and 'height'.
        Each row is processed to calculate BMI and CVD risk, and the
        resulting data is categorized and stored in the `self.data` attribute.

        Args:
            filename (str): The path to the input CSV file.
        """
        with open(filename) as f:
            for row in csv.DictReader(f):
                subject = {
                    "subject_id": row["id"],
                    "age": int(row["age"]),
                    "weight": float(row["weight"]),
                    "height": float(row["height"]),
                }

                subject["bmi"] = self.calculate_bmi(
                    subject["weight"], subject["height"]
                )
                subject["cvd_prob"] = self.calculate_cvd_risk(
                    subject["bmi"], subject["age"]
                )

                category = self.categorize_bmi(subject["bmi"])
                self.data[category].append(json.dumps(subject))

        print(f"Processed {sum(len(v) for v in self.data.values())} subjects")

    def save_json(self, filename="data/labeled_data.json"):
        """
        Saves the processed data dictionary to a JSON file.

        Args:
            filename (str): The path to the output JSON file.
        """
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=2)

    def print_stats(self):
        """
        Prints statistics on the processed dataset.

        Displays the total number of subjects and the count and percentage
        of subjects in each BMI category.
        """
        total = sum(len(v) for v in self.data.values())
        for category, subjects in self.data.items():
            count = len(subjects)
            percent = (count / total * 100) if total > 0 else 0
            print(f"{category}: {count} ({percent:.1f}%)")
