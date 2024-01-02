import pickle
import matplotlib.pyplot as plt


#data_dir='../../data/model_devel_files/Scan_Data/L5/'
data_dir = ''
config_dir='../../config/'
graph = True
def average_dict_val(dict):
    lower_limit = -20
    upper_limit = 5

    sum = 0
    num_of_samples = 0

    for entry in dict:
        if  lower_limit < entry < upper_limit:
            sum +=dict[entry]
            num_of_samples+=1
    if num_of_samples ==0:
        avg = 0
    else:
        avg = sum/num_of_samples
    print("layer :", layer)
    print("AVG: ", avg)
    print("------------------------")

    return avg

all_weld_width = pickle.load(open(data_dir+'all_welds_width.pickle', 'rb'))
average_values = []
layers = []

for layer in all_weld_width[0].keys():
    average = average_dict_val(all_weld_width[0][layer])
    average_values.append(average)
    layers.append(layer)
    if graph:
        plt.plot(all_weld_width[0][layer].keys(), all_weld_width[0][layer].values())
        plt.title(layer)
        
plt.show()
maximum = max(average_values)

highest_vals = sorted(average_values, reverse=True)[:3]

highest_average = sum(highest_vals)/3

print ("maximum width: ", maximum)
print ("highest 3: ", highest_average)