#!/bin/bash

for i in {1..22}
do
	for j in {1..22}
	do
	k=$(( 2048 * 2048 ))
	divisor=$(( (2 ** ($i * 2)) * (2 ** ($j * 2)) ))
	if [ $divisor -lt $k ] && [ $divisor -gt 0 ];
	then
		k=$(( $k / $divisor ))
		if [ $k -gt 0 ]
		then
			k=$(echo "$k" | awk '{print sqrt($1)}')
			if [[ $k =~ ^[0-9]+$ ]]
			then
				i2=$(( 2 ** $i ))
				j2=$(( 2 ** $j ))
				echo $i2
				echo $j2
				echo $k
				rm -f jacobiRWD
				make jacobiRWD N=2048 NT=$i2 NB=$j2 NK=$k
				#//nvcc -Isrc/ -DN=2048 -DNT=$i2 -DNB=$j2 -DNK=$k obj/JacobiRWD.o obj/mainRWD.o -o jacobiRWD
				srun -N1 --gres=gpu:4 jacobiRWD
				echo
			fi
		fi
	fi
	done
done
