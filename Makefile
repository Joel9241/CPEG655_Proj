CC=gcc
CCFLAGS=-I$(SRCDIR)/ -DN=2048
GC=nvcc
GCFLAGS=-I$(SRCDIR)/ -DN=2048 -DNT=2 -DNB=16 -DNK=64
OBJDIR=obj
SRCDIR=src
TESTDIR=tests

all: jacobi jacobiRWD cpuTests rwdTests

jacobi: $(OBJDIR)/Jacobi.o $(OBJDIR)/main.o
	$(CC) $(CCFLAGS) $(OBJDIR)/Jacobi.o $(OBJDIR)/main.o -o jacobi

jacobiRWD: $(OBJDIR)/JacobiRWD.o $(OBJDIR)/mainRWD.o
	$(GC) $(GCFLAGS) $(OBJDIR)/JacobiRWD.o $(OBJDIR)/mainRWD.o -o jacobiRWD

$(OBJDIR)/Jacobi.o: $(SRCDIR)/Jacobi.h $(SRCDIR)/Jacobi.c
	$(CC) $(CCFLAGS) -c $(SRCDIR)/Jacobi.c
	mv Jacobi.o $(OBJDIR)/

$(OBJDIR)/main.o: $(SRCDIR)/main.c
	$(CC) $(CCFLAGS) -c $(SRCDIR)/main.c
	mv main.o $(OBJDIR)/	

$(OBJDIR)/mainRWD.o: $(SRCDIR)/mainRWD.cu
	$(GC) $(GCFLAGS) -c $(SRCDIR)/mainRWD.cu
	mv mainRWD.o $(OBJDIR)/	

cpuTests: $(OBJDIR)/JacobiTests.o $(OBJDIR)/Jacobi.o
	$(CC) $(CCFLAGS) $(OBJDIR)/JacobiTests.o $(OBJDIR)/Jacobi.o -o cpuTests

$(OBJDIR)/JacobiTests.o: $(SRCDIR)/Jacobi.h $(TESTDIR)/JacobiTests.c
	$(CC) $(CCFLAGS) -c $(TESTDIR)/JacobiTests.c
	mv JacobiTests.o $(OBJDIR)/

rwdTests: $(OBJDIR)/JacobiRWDTests.o $(OBJDIR)/JacobiRWD.o
	$(GC) $(GCFLAGS) $(OBJDIR)/JacobiRWDTests.o $(OBJDIR)/JacobiRWD.o -o rwdTests

$(OBJDIR)/JacobiRWDTests.o: $(SRCDIR)/JacobiRWD.cuh $(TESTDIR)/JacobiRWDTests.cu
	$(GC) $(GCFLAGS) -c $(TESTDIR)/JacobiRWDTests.cu
	mv JacobiRWDTests.o $(OBJDIR)/

$(OBJDIR)/JacobiRWD.o: $(SRCDIR)/JacobiRWD.cuh $(SRCDIR)/JacobiRWD.cu
	$(GC) $(GCFLAGS) -c $(SRCDIR)/JacobiRWD.cu
	mv JacobiRWD.o $(OBJDIR)/

clean:
	rm -rf $(OBJDIR)/*.o jacobi rwdTests cpuTests
