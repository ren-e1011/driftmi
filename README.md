# Syclop-MINE
A MINE based application to assess Syclop 

## Needed packages:

### pytorch: version - 1.4.0 

pip install torch torchvision

### pandas: version - 0.25.1 latest #Officially Python 3.6.1 and above, 3.7, and 3.8

pip install pandas

### pyprind: version - 2.11.2

pip install PyPrind


# sys.argv gets 4 values:

[1] - the optimizer - one of three Adam, SGD, RMSprop

[2] - lr

[3] - batch_size

[4] - epochs

[5] - train true/false 1/0

[6] - Net num - 1-3

[7] - number_descending_blocks - i.e. how many steps should the fc networks take,starting at 2048 nodes and every step divide the size by 2. max number is 7                                

[8] - number_repeating_blocks - the number of times to repeat a particular layer

[9] - repeating_blockd_size - the size of nodes of the layer to repeat

For example, number_descending_blocks = 5, number_repeating_blocks = 3, and repeating_blockd_size = 512 then the net architecture will look like:

statistical_estimator_DCGAN_3(

(conv1): Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(elu): ELU(alpha=1.0, inplace=True)

(conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(conv3): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(conv12): Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(conv22): Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(conv32): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

(fc): ModuleList(

(0): Linear(in_features=2048, out_features=1024, bias=True)

(1): Linear(in_features=1024, out_features=512, bias=True)

(2): Linear(in_features=512, out_features=512, bias=True)

(3): Linear(in_features=512, out_features=512, bias=True)

(4): Linear(in_features=512, out_features=512, bias=True)

(5): Linear(in_features=256, out_features=128, bias=True)

(6): Linear(in_features=128, out_features=64, bias=True)

)

(fc_last): Linear(in_features=64, out_features=1, bias=True)
)

