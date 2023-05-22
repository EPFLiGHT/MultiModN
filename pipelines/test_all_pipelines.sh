#!/bin/bash

parent_path=$(dirname $(realpath $0))

for pipeline in $(find $parent_path -name '*pipeline.py')
do
    python3 $pipeline -e 5 -m false -y false -p false -r false
done
