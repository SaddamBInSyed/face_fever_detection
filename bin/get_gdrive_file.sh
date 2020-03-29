#!/bin/bash

# example
# bash ./get_gdrive_file.sh 1DUOzNlCQlCjBPKaAyf8YQCCPrq6gaWfv tmp.txt
if [ $# -ne 2 ]
then
	echo "ERROR: Expected exactly 2 arguments. got $#"
	echo "USAGE: $0 <google-id> <destination-file>"
	exit 1
fi

ggID=$1  
ggURL='https://drive.google.com/uc?export=download'  
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
filename=$2
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  

echo google-id:           $ggID
echo destination-file:    $2
echo google coockie code: $getcode
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o $2
