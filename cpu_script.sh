#!/bin/bash

idx=1
for i in {1..2}
do
    start=256
    end=16384
    batch=4
    for j in {1..1}
    do
        printf "batch number: %d" $batch
        curr_dt=$(date "+%Y.%m.%d-%H.%M.%S")
        path='./batchdata/'$curr_dt
        python3 ./Testing.py \
        --size=$batch \
        --mode="batch"\
        --Start_size=$start\
        --End_size=$end \
        --Loop_NUM=$i \
        --Store_path=$path \
        --Test="customized" \
        --Running_platform="CPU"
        if test -f "memoryfile.txt"; then
            echo "memoryfile.txt exists."
            mv "memoryfile.txt" $HOME"/flat/batchdata/"$curr_dt
        fi
        cd ./batchdata
        new_name="file-$idx-256-16k-batchtest"
        mv $curr_dt $new_name
        idx=$((idx+1))
        #head=$((head+1))
        cd ..
    done
done
echo "finish"
