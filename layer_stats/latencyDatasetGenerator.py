from LatencySampler import getLayerwiselatency
from layer_stats.utils.operations import database_spawn
from layer_stats.utils.checker import get_model_list

'''
To change the number of data samples collected per image, change the parameters defined in `LatencySampler.py` script.
'''

op_executor = database_spawn(preprocess=True)

# Define the device for the evaluation
Device = "cpu"

# Define sample paths 
sample_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

model_list = get_model_list()

for model_name in model_list:
    print(f"Latency eveluation for `{model_name}`")

    # Run power monitoring and create datasets for the given image set
    getLayerwiselatency(op_executor,model_name,Device,sample_paths)
