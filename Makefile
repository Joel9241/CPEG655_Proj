CC=gcc
CCFLAGS=-DN=2

all: jacobi test

jacobi: Jacobi.o main.o
	$(CC) $(CCFLAGS) Jacobi.o main.o -o jacobi

Jacobi.o: Jacobi.h Jacobi.c
	$(CC) $(CCFLAGS) -c Jacobi.c

main.o: main.c
	$(CC) $(CCFLAGS) -c main.c

test: JacobiTests.o Jacobi.o
	$(CC) $(CCFLAGS) JacobiTests.o Jacobi.o -o tests

JacobiTests.o: Jacobi.h JacobiTests.c
	$(CC) $(CCFLAGS) -c JacobiTests.c

clean:
	rm -rf *.o jacobi tests
