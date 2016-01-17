import csv

class CSVWriter:

    def __init__(self, matchers, csv_file):
        self.matchers = matchers
        self.csv_file = csv_file

    def write_header(self):
        with open(self.csv_file, 'w') as f:
            csvwriter = csv.writer(f)

            csv_header = [
                'FeatureExtractor/Descriptor',
                'Image 1 Keypoints',
                'Image 1 Keypoints Time',
                'Image 1 Descriptions',
                'Image 1 Descriptions Time',
                'Image 2 Keypoints',
                'Image 2 Keypoints Time',
                'Image 2 Descriptions',
                'Image 2 Descriptions Time',
            ]

            for m in self.matchers:
                csv_header.append('%s Inlier' % m.name)
                csv_header.append('%s Time' % m.name)
                csv_header.append('%s Outlier' % m.name)

                csv_header.append('%s Knn Inlier' % m.name)
                csv_header.append('%s Knn Time' % m.name)
                csv_header.append('%s Knn Outlier' % m.name)
            csvwriter.writerow(csv_header)

    def write_column(self, column):
        with open(self.csv_file, 'a') as f:
            f.write(str(column))
            f.write(',')

    def write_row(self, row=''):
        with open(self.csv_file, 'a') as f:
            if row:
                f.write(str(row))
            f.write('\n')
