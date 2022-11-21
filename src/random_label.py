import csv
import os
import random

LABELS = ["calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def main():
    random.seed()

    with open("results.csv", "w+") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(["filename", "label"])

        test_path = os.path.join(os.getcwd(), "..", "test_data")
        files = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
        for filename in files:
            if (os.path.splitext(filename)[1] == ".wav"):
                writer.writerow([os.path.splitext(filename)[0], random.choice(LABELS)])


if __name__ == "__main__":
    main()