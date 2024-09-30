import csv

def create_csv(file_path, contents):
    rows = zip(contents)
    with open(file_path, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerows(rows)

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        csv_read = csv.reader(f)
        for c in csv_read:
            c = c[0].split(',')
            data.append(float(c[1].split(')')[0]))
    return data 

