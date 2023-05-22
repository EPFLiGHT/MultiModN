#!/bin/bash

URL="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
dir="/data/titanic/"
source_filename="titanic.csv"

parent_path=$(dirname $(realpath $0))
project_path="$(dirname $(dirname $parent_path))"
data_path="$project_path$dir"

mkdir -p $data_path

if [ ! -f $data_path$source_filename ]
then wget $URL -O $data_path$source_filename
fi