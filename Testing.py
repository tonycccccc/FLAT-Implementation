#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob2
import time
import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd



# Revise the FlatDataflow function to make it clearer

def FlatDataflow(loop_num, query, key, value, bias, batch_granularity_level=1, head_granularity_level=8, length_granularity_level=64,
            batchTrue=False, headTrue=False, lengthTrue=True):
  if (tf.config.list_physical_devices('CPU') or tf.config.list_physical_devices('GPU')):
    if (os.environ['CUDA_VISIBLE_DEVICES'] == '0'):
      tf.config.experimental.reset_memory_stats('GPU:0')
    # If the dimension of the input tensor is 3 rather than 4, expand its dimension for the batch
    if (len(query.shape) == 3):
      query = query[None, :, :, :]
      key = key[None, :, :, :]
      value = value[None, :, :, :]  
    batch_size, source_length, head_num, dim = tf.shape(query).numpy()
    memory=0
    # Set a fixed bias value here
    bias_value = bias
    for batch in tf.range(0, batch_size, batch_granularity_level):
      # The lowest granularity is batch level.
      if (batch_granularity_level != 1 or batchTrue):
        end_batch = batch + batch_granularity_level if batch + batch_granularity_level <= batch_size else batch_size
        query_source = tf.gather(query[:, :, :, :], indices=tf.range(batch, end_batch), axis=0)
        key_source = tf.gather(key[:, :, :, :], indices=tf.range(batch, end_batch), axis=0)
        value_source = tf.gather(value[:, :, :, :], indices=tf.range(batch, end_batch), axis=0)
        result = tf.einsum("BTNH, BFNH->BNFT", key_source, query_source)
        result += bias_value
        result = tf.nn.softmax(result, name="attention_weights")
        result = tf.nn.dropout(result, rate=0.4)
        attention_output = tf.einsum("BNFT,BTNH->BFNH", result, value_source)
      else:
        for head in tf.range(0, head_num, head_granularity_level):
          # The lowest granularity is head level.
          if (head_granularity_level != 1 or headTrue):
            end_head = head + head_granularity_level if head + head_granularity_level <= head_num else head_num
            query_source = tf.gather(query[batch, :, :, :], indices=tf.range(head, end_head), axis=1)
            key_source = tf.gather(key[batch, :, :, :], indices=tf.range(head, end_head), axis=1)
            value_source = tf.gather(value[batch, :, :, :], indices=tf.range(head, end_head), axis=1)
            result = tf.einsum("TNH, FNH->NFT", key_source, query_source)
            result += bias_value
            result = tf.nn.softmax(result, name="attention_weights")
            result = tf.nn.dropout(result, rate=0.4)
            logit = tf.einsum("NFT,TNH->FNH", result, value_source)
            if head == 0:
              attention_output = logit
            else:
              attention_output = tf.concat([attention_output, logit], axis=1)
          else:
            #Lowest granularity is length level
            for length in tf.range(0, source_length, length_granularity_level):
              end_length = length + length_granularity_level if length + length_granularity_level <= source_length else source_length
              query_source = tf.gather(query[batch, :, head, :], indices=tf.range(length, end_length), axis=0)
              key_source = key[batch, :, head, :]
              result = tf.einsum("TH, FH->FT", key_source, query_source)
              result += bias_value
              result = tf.nn.softmax(result, name="attention_weights")
              result = tf.nn.dropout(result, rate=0.4)
              if length == 0:
                lengthOutput = result
              else:
                lengthOutput = tf.concat([lengthOutput, result], axis=0)
            value_source = value[batch, :, head, :]
            lengthRes = tf.einsum("FT,TH->FH", lengthOutput, value_source)
            lengthRes = tf.expand_dims(lengthRes, axis=1)
            if (head == 0):
              attention_output = lengthRes
            else:
              attention_output = tf.concat([attention_output, lengthRes], axis=1)
        attention_output = tf.expand_dims(attention_output, axis=0)
      if (batch == 0):
        output = attention_output
      else:
        output = tf.concat([output, attention_output], axis=0)
    stoptime = time.time()
    if (os.environ['CUDA_VISIBLE_DEVICES'] == '0'):
      memoryinfo = tf.config.experimental.get_memory_info('GPU:0')
      memory = memoryinfo['peak']
    return (0, 0, memory, 0, stoptime)


def CallFLAT(batch_usage, batch_tile, head_usage, head_tile, length_usage, length_tile, test_dimension, test_shape,
            start_size, end_size, loop_num, path, head_per_dimension):
  # Read in all the files
  query_file = glob2.glob("/nethome/zchen752/flat/FILES/logging_query*.txt")
  key_file = glob2.glob("/nethome/zchen752/flat/FILES/logging_key*.txt")
  value_file = glob2.glob("/nethome/zchen752/flat/FILES/logging_value*.txt")

  # Randomly set a bias value for now
  BIAS = 0.02

  running_time = []

  FILENUM = len(query_file)
  BATCHSIZE = 64
  queryin = []
  keyin = []
  valuein = []
  for file in query_file:
      qfile = tf.io.read_file(file)
      query = tf.io.parse_tensor(qfile, out_type=tf.float32)
      query = tf.gather(query, indices=tf.range(0,head_per_dimension), axis=-1)
      queryin.append(query)
  for file in key_file:
      kfile = tf.io.read_file(file)
      key = tf.io.parse_tensor(kfile, out_type=tf.float32)
      key = tf.gather(key, indices=tf.range(0,head_per_dimension), axis=-1)
      keyin.append(key)
  for file in value_file:
      vfile = tf.io.read_file(file)
      value = tf.io.parse_tensor(vfile, out_type=tf.float32)
      value = tf.gather(value, indices=tf.range(0,head_per_dimension), axis=-1)
      valuein.append(value)
  queryin = tf.stack(queryin)
  keyin = tf.stack(keyin)
  valuein = tf.stack(valuein)
  print("Files Successfully Loaded!")
  # Set up the parameters
  batch = batch_tile
  head = head_tile
  length = length_tile
  batchTrue = batch_usage
  headTrue = head_usage
  lengthTrue = length_usage

  peak_memory=[]
  #batch_shape, length_shape, num_shape, head_shape = test_shape[0], test_shape[1], test_shape[2], test_shape[3]
  counter_i = 0
  # Length test mode: length increase from 256 to 4096
  if (test_dimension == "length"):
    fileidx = np.random.randint(FILENUM)
    batchidx = np.random.randint(BATCHSIZE)
    query = queryin[fileidx][batchidx, :, :, :]
    key = keyin[fileidx][batchidx, :, :, :]
    value = valuein[fileidx][batchidx, :, :, :]

    # Generate start matrix with shape 1 * 256 * 16 * 64
    # Randomly pick a file number
    for i in range(start_size // 64 - 1):
      fileidx = np.random.randint(FILENUM)
      batchidx = np.random.randint(BATCHSIZE)
      query = tf.concat((query, queryin[fileidx][batchidx, :, :, :]), axis=0)
      key = tf.concat((key, keyin[fileidx][batchidx, :, :, :]), axis=0)
      value = tf.concat((value, valuein[fileidx][batchidx, :, :, :]), axis=0)

    for idx in range(start_size // 64, end_size // 64, 1):
      start_time = time.time()
      peakOld, currOld, peakCurr, currCurr, stoptime = FlatDataflow(idx-start_size//64, query, key, value, BIAS, batch_granularity_level=batch, 
                                                      head_granularity_level=head, length_granularity_level=length, batchTrue=batchTrue, headTrue=headTrue, lengthTrue=lengthTrue)
      running_time.append(stoptime - start_time)
      peak_memory.append(peakCurr)
      fileidx = np.random.randint(FILENUM)
      batchidx = np.random.randint(BATCHSIZE)

      query = tf.concat((query, queryin[fileidx][batchidx, :, :, :]), axis=0)
      key = tf.concat((key, keyin[fileidx][batchidx, :, :, :]), axis=0)
      value = tf.concat((value, valuein[fileidx][batchidx, :, :, :]), axis=0)
      print("LOOP %d" % (idx) )
      if (counter_i + 1 == 25):
        counter_i = 0
        if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
          cmd = "top -b -n 1|head -n 20 >> memoryfile.txt"
        else:
          cmd = "nvidia-smi >> memoryfile.txt"
        os.system(cmd)
      counter_i = counter_i + 1


  # Batch test mode: batch increases from 256 to 4096

  elif (test_dimension == "batch"):
    fileidx = np.random.randint(FILENUM)
    query = queryin[fileidx][:, :, :, :]
    key = keyin[fileidx][:, :, :, :]
    value = valuein[fileidx][:, :, :, :]

    for i in range(start_size // 64 - 1):
      fileidx = np.random.randint(FILENUM)
      query = tf.concat((query, queryin[fileidx][:, :, :, :]), axis=0)
      key = tf.concat((key, keyin[fileidx][:, :, :, :]), axis=0)
      value = tf.concat((value, valuein[fileidx][:, :, :, :]), axis=0)

    for idx in range(start_size//64, end_size//64, 1):
      start_time = time.time()
      peakOld, currOld, peakCurr, currCurr, stoptime = FlatDataflow(idx-start_size//64, query, key, value, BIAS, batch_granularity_level=batch, 
                                                      head_granularity_level=head, length_granularity_level=length, batchTrue=batchTrue, headTrue=headTrue, lengthTrue=lengthTrue)
      running_time.append(stoptime - start_time)
      peak_memory.append(peakCurr)
      fileidx = np.random.randint(FILENUM)
      query = tf.concat((query, queryin[fileidx][:, :, :, :]), axis=0)
      key = tf.concat((key, keyin[fileidx][:, :, :, :]), axis=0)
      value = tf.concat((value, valuein[fileidx][:, :, :, :]), axis=0)
      print("LOOP %d" % (idx) )
      if (counter_i + 1 == 25):
        counter_i = 0
        if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
          cmd = "top -b -n 1|head -n 20 >> memoryfile.txt"
        else:
          cmd = "nvidia-smi >> memoryfile.txt"
        os.system(cmd)
      counter_i = counter_i + 1

  # Customized test option
  elif (test_dimension == "customized"):
    fileidx = np.random.randint(FILENUM)
    query = queryin[fileidx][:, :, :, :]
    key = keyin[fileidx][:, :, :, :]
    value = valuein[fileidx][:, :, :, :]
    for i in range(start_size // 64 - 1):
      fileidx = np.random.randint(FILENUM)
      query = tf.concat((query, queryin[fileidx][:, :, :, :]), axis=1)
      key = tf.concat((key, keyin[fileidx][:, :, :, :]), axis=1)
      value = tf.concat((value, valuein[fileidx][:, :, :, :]), axis=1)
    for idx in range(start_size // 64, end_size // 64, 1):
      start_time = time.time()
      peakOld, currOld, peakCurr, currCurr, stoptime = FlatDataflow(idx-start_size//64, query, key, value, BIAS, batch_granularity_level=batch, 
                                                      head_granularity_level=head, length_granularity_level=length, batchTrue=batchTrue, headTrue=headTrue, lengthTrue=lengthTrue)
      running_time.append(stoptime - start_time)
      peak_memory.append(peakCurr)
      fileidx = np.random.randint(FILENUM)
      batchidx = np.random.randint(BATCHSIZE)

      query = tf.concat((query, queryin[fileidx][:, :, :, :]), axis=1)
      key = tf.concat((key, keyin[fileidx][:, :, :, :]), axis=1)
      value = tf.concat((value, valuein[fileidx][:, :, :, :]), axis=1)
      print("LOOP %d" % (idx) )
      if (counter_i + 1 == 25):
        counter_i = 0
        if os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
          cmd = "top -b -n 1|head -n 20 >> memoryfile.txt"
        else:
          cmd = "nvidia-smi >> memoryfile.txt"
        os.system(cmd)
      counter_i = counter_i + 1


  timefilename=path+"/data"+("{0}".format(loop_num))+".txt"
  memoryfilename=path+"/memory"+("{0}".format(loop_num))+".txt"
  with open(timefilename, 'w') as f:
    for item in running_time:
      f.write(str(item) + "\n")
    f.close()
  if (os.environ['CUDA_VISIBLE_DEVICES'] == '0'):
    with open(memoryfilename, 'w') as f:
      for item in peak_memory:
        f.write(str(item) + "\n")
      f.close()
  print("Finished!")



# Code block building graph with x-axis as sequence length and y-axis as peak memory usage

if __name__=="__main__":
  parser = argparse.ArgumentParser(description="Enter the Granularity Level for FLAT")
  parser.add_argument('--size', type=int, default=1, help='Granularity')
  parser.add_argument('--mode', type=str, default="length", help="Choose FLAT mode -- batch, head or length")
  parser.add_argument('--TestShape', type=int, nargs="+", default=[1, 256, 16, 64], help="Starting shape of matrix")
  parser.add_argument('--Start_size', type=int, default=256, help="Start size of test dimension")
  parser.add_argument('--End_size', type=int, default=4096, help="Start size of test dimension")
  parser.add_argument('--Test', type=str, default='head', help="Choose one dimension to test")
  parser.add_argument('--Loop_NUM', type=int, default=0)
  parser.add_argument('--Store_path', type=str, default="")
  parser.add_argument('--Running_platform', type=str, default="CPU")
  parser.add_argument('--Head_Per_Dimension', type=int, default=64)
  args = parser.parse_args()
  if (args.Running_platform == "CPU"):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  batch_usage = False
  head_usage = False
  length_usage = False
  batch_size = 1
  head_size = 1
  length_size = 1
  os.makedirs(args.Store_path, exist_ok=True)
  if (args.mode == "batch"):
    batch_usage = True
    batch_size = args.size
  elif (args.mode == "head"):
    head_usage = True
    head_size = args.size
  else:
    length_usage = True
    length_size = args.size
  CallFLAT(batch_usage, batch_size, head_usage, head_size, length_usage, length_size, args.Test, args.TestShape,
            args.Start_size, args.End_size, args.Loop_NUM, args.Store_path, args.Head_Per_Dimension)



