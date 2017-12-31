DEFAULT_BOUND=1000000

if [ $# -eq 0 ]
then
	echo Give a json path as parameter
	exit
fi
if [ $# -eq 1 ]
then
	bound=$DEFAULT_BOUND
	echo Defaut upper bound is: $bound
else
	bound=$2
fi
nb_steps=$(sed 's/.*\"value\": \([0-9]*\), \"update\": true.*/\1/' "$1")
while [ $nb_steps -le $bound ]
do
	python3 main.py "$1" --train
done
