#!/bin/bash

# Simple script that cleans tex directory of unnecessary files for git-pushing 

texdir="/home/dgabutler/Work/CMEEProject/Code/tex"

rm "$texdir"/*.gz
rm "$texdir"/*.aux
rm "$texdir"/*.log

