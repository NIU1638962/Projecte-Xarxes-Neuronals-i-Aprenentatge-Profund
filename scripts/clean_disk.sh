#!/bin/bash

echo 'Space used before clean:'

du -hs ~

conda clean --all -y

cd ~/.local/share/Trash/

rm -rfv *

cd ~/.cache/

rm -rfv *

echo 'Space used after clean:'

du -hs ~
