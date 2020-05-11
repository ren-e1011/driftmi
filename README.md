# Syclop-MINE
A MINE based application to assess Syclop 

## Needed packages:

### pytorch: version - 1.4.0 

pip install torch torchvision

### pandas: version - 0.25.1 latest #Officially Python 3.6.1 and above, 3.7, and 3.8

pip install pandas

### pyprind: version - 2.11.2

pip install PyPrind


Run MINE on terminal:​

mine_run.py 2 0.003 500 200 1 2 combined 3 2 2 512 6 different​

sys.argv gets 13 values:​
[1] - the optimizer - one of three (1) - SGD,(2) - Adam, (3) - RMSprop​

[2] - lr  - the combined same option diverged when lr was 3e-3. ​

[3] - batch_size​

[4] - epochs​

[5] - train true/false 1/0​

[6] - Net num - 1-3​

[7] - traject/MNIST/combined - What form of mine to compute​

[8] - number_descending_blocks - i.e. how many steps should the fc networks take, starting at 2048 nodes and every step 
divide the size by 2. max number is 7                                ​

[9] - number_repeating_blocks - the number of times to repeat a particular layer​

[10] - repeating_blockd_size - the size of nodes of the layer to repeat​

[11] - traject_max_depth​

[12] - traject_num_layers​

[13] - same/different minist/trajectory - to use the same image that the trajectory ran on as the joint distribution.​

​

An example of a run:​

bsub -q gpu-short -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3 -gpu num=4:j_exclusive=no -R "select[mem>8000] rusage[mem=8000]" -o out.%J -e err.%J python3 mine_run.py 2 0.0003 500 200 1 2 combined 3 2 2 512 6 same     ​

 Run Trajectories classification on terminal:​

syclop_classification_gpu.py 2 0.003 500 150 0 0.5 0.2 5 [3,1] [1,2] 512 5 4 0 0​

sys.argv gets 15 values:​

 [1] - the optimizer - one of three Adam, SGD, RMSprop​

 [2] - lr​

 [3] - batch_size​

 [4] - epochs​

 [5] - Net num - 1-3​

 [6] - Dropout fc​

 [7] - Dropout conv​

 [8] - kernel​

 [9] - stride​

 [10] - pooling​

 [11] - max_depth​

 [12] - num_layers                           ​

 [13] - reapeating_block_depth - what number in num_layers should we repeat​

 [14] - repeating_block_number - how many times to repeat​

 [15] - data_path - if the data is stored outside the working directory. If not sys.argv[0] should be 0​

Example of a run (run with best results got 85.7 with these paremeters ):​

bsub -q gpu-short -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3 -gpu num=2:j_exclusive=no -R "select[mem>4500] rusage[mem=4500]" -o out.%J -e err.%J python3 syclop_classification_gpu.py 2 0.003 500 150 0 0.5 0.1 5 [3,1] [1,2] 128 5 4 0 0
