import fabric
import pathlib
from file_transfer import SCP, Rsync


source_base = '/home/ugiri/sigmafnl/datasets/bimodal_dataset'
source_list = [f'{source_base}/s8_{sigma}/{i}/matter_patches_{i}_512_256_256.npy'
                        for i in range(90,500) for sigma in ['p', 'm']]

destination_base = '/gpfs/ugiri/sigma-fnl-dataset/bimodal_dataset'
destination_list = [f'{destination_base}/s8_{sigma}/{i}/matter_patches_{i}_512_256_256.npy'
                        for i in range(90, 500) for sigma in ['p', 'm']]


client = SCP()
client.start_transfer(source_list=source_list, destination_list=destination_list)


