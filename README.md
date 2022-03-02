# FLAT-Implementation
Implementation of Fused Logit Attention Tiling

This github repository contains the FLAT (Fused Logit Attention Tiling) implementation and FLAT1 simulator profiling code's runtime and memory performance. Users can test different granularities here with input data under ./FILES directory and get the clear statistics in ./data directory.

# How to start?
1. Make sure install all the required libraries, which includes Tensorflow2.0, Numpy, glob2 and argparse.
2. Put the query, key, and value matrix file under the ./FILES directory. User can extract the data through tf.io by running the official transformer model. Here we just put several files as samples.
3. Edit the script. We provide a bunch of arguments in the script file. Understanding them is crucial foer using the simulator. Use -h to check all the flags.
      - --mode : Choose FLAT granularity mode, which are **batch**, **head**, and **length**
      - --size : Size of one-time processing, e.g how many batches / heads to perform
      - --Start_size : Starting size of the test dimension
      - --End_size : Ending size of the test dimension
      - --Test : Testing dimension. 
        - User can select length as testing dimension, starting from length 256 to 16k, and use batch = 4 as granularity
      - --Store_path : Path to store the memory and running time data
      - --Running_platform : Platform to run -- user can select either CPU or GPU. For GPU, it will use the first one detected(GPU 0)
      - --Loop_NUM : index of loop retition number
4. Check the result data under the specified path. It should contain data.txt and memoryfile.txt, where each line in data.txt records the running time of the test case and memoryfile.txt traces memory usage snapshot. **Notice sometimes memoryfile.txt might not exist since we only take snapshots every 20 loops**.
      
# How we track the profiling data?
1. Runtime data:
    - For runtime, we use the simplest method, subtracting the start time from stop time and get the difference. Basically, for each loop FLAT processes, it tracks       the time it takes and store it under the corresponding director under ./data.
2. Memory usage:
    - For memory profiling on CPU, we take the snapshot of the htop command under linux system to dynmaically track the status of memory usage.
    - For memory profiling on GPU, we do two strategies. First one is to take snapshot of nvidia-smi command report. Second one is to use tf.config.memory_status()       to track the peak and current memory usage. Memory profiling data will also be stored in the same directory under ./data.

# Not having a GPU locally?
Besides the local testing method, we also provide a colab version of FLAT. User might want to store the data file in google drive and mount it to the colab. Similar working process is needed here.

# Running FLAT on TPU?
We provide a separate file running FLAT on TPU. Check the notebook or even modify it if you are not satisfied.

# Challenges:
There are many leftover issues right now. For example, GPU testing often crashes due to large intermediate footprint. CPU testing are not stable and always has some weird glitches. We are working on the solution now.

# What to expect?
We will be using more visualization tools to visualize memory usage and deeply understand what is going on inside the FLAT. We will also try to add some HLS taste to the project and implement it on FPGA to see if we can make FLAT even faster.
