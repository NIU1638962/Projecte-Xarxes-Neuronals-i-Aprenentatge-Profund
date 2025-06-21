#!/bin/bash
bash $1 -b -u -p $2
source ¤3/bin/activate
conda init --all
conda create --file $3