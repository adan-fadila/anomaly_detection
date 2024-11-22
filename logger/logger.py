import csv


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def append_data_to_csv(row_data, filename):
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row_data['data'].values())
