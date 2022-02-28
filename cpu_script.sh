#!/bin/bash

idx=1
#How many times to repeat this test
for i in {1..2}
do
    start=256
    end=16384
    head=1
    #Increase number from 1 to 16
    for j in {1..16}
    do
        printf "head number: %d" $head
        curr_dt=$(date "+%Y.%m.%d-%H.%M.%S")
        path='./data/'$curr_dt
        python3 ./Testing.py \
        --size=$head \
        --mode="head"\
        --Start_size=$start\
        --End_size=$end \
        --Loop_NUM=$i \
        --Store_path=$path \
        --Test="length" \
        --Running_platform="CPU"
        if test -f "memoryfile.txt"; then
            echo "memoryfile.txt exists."
            mv "memoryfile.txt" $HOME"/flat/batchdata/"$curr_dt
        fi
        cd ./data
        new_name="file-$idx-256-16k-headtest"
        mv $curr_dt $new_name
        idx=$((idx+1))
        head=$((head+1))
        cd ..
    done
done
echo "finish"
