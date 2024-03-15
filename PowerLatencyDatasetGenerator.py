from PowerLatencySampler import getLayerwisePowerLatency
from operations import database_spawn
from checker import get_model_list

'''
To change the number of data samples collected per image, change the parameters defined in `PowerLatencySampler.py` script.
'''

op_executor = database_spawn(preprocess=True)

# Define the device for the evaluation
Device = "cpu"

# Define sample paths 
sample_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

model_list = get_model_list()

for model_name in model_list:

    # Run power monitoring and create datasets fro the given image set
    getLayerwisePowerLatency(op_executor,model_name,Device,sample_paths)
