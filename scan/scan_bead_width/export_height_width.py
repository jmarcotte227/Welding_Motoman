import numpy as np
import sys
import csv
sys.path.append('../../weld')
import weld_dh2v 
import weld_w2v 

with open('height_width_data.csv','w', newline='') as file:
    writer=csv.writer(file)

    for vel in range(5,16):
        row = [weld_dh2v.v2dh_loglog(vel, 160, 'ER_4043'), weld_w2v.v2w_loglog(vel, 160, 'ER_4043')]
        writer.writerow(row)
        print(row)

        