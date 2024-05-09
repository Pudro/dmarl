#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [directory] [file name prefix] [target number of files]"
    echo "This script assumes the model files are named <filename>_<number>.tar and copies the models sequentially, until the target number of models is reached."
    exit 1
fi

directory=$1
prefix=$2
target=$3
counter=0

while [ $counter -lt $target ]; do
    for file in "$directory"/"$prefix"_*.tar; do
        cp "$file" "$directory"/"$prefix"_"$counter".tar
        counter=$((counter+1))
        if [ $counter -ge $target ]; then
            break
        fi
    done
done

# You can save this script to a file, for example, `copy_files.sh`, and then make it executable using the command `chmod +x copy_files.sh`. To run the script, you would provide the directory, file name prefix, and target number of files as arguments like this:

# ```bash
# ./copy_files.sh /path/to/directory prefix 7
# ```

# This script will copy files with the specified prefix in their names from the directory into the same directory, creating new files with consecutive suffixes until the target number of files is reached.
