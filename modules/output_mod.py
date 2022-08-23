import csv

class output(object):
#データを書き出すクラス。今はcsv用しかない
    def __init__(self, output_name, data, header = "", header_switch = False):
        self.output_name = output_name
        self.data = data
        self.header_switch = header_switch
        self.header = header
    
    def output_csv(self):
        with open (self.output_name, 'w') as fname:
            writer = csv.writer(fname)
            if self.header_switch:
                writer.writerow(self.header)
            for i in range(len(self.data)):
                writer.writerow(self.data[i])