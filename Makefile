CC=gcc
CCFLAGS=-I$(SRCDIR)/ -DN=2
GC=nvcc
GCFLAGS=-I$(SRCDIR)/ -DN=2 -DNT=1 -DNB=1
OBJDIR=obj
SRCDIR=src
TESTDIR=tests

all: jacobi cpuTests rwdTests

jacobi: $(OBJDIR)/Jacobi.o $(OBJDIR)/main.o
	$(CC) $(CCFLAGS) $(OBJDIR)/Jacobi.o $(OBJDIR)/main.o -o jacobi

$(OBJDIR)/Jacobi.o: $(SRCDIR)/Jacobi.h $(SRCDIR)/Jacobi.c
	$(CC) $(CCFLAGS) -c $(SRCDIR)/Jacobi.c
	mv Jacobi.o $(OBJDIR)/

$(OBJDIR)/main.o: $(SRCDIR)/main.c
	$(CC) $(CCFLAGS) -c $(SRCDIR)/main.c
	mv main.o $(OBJDIR)/	

cpuTests: $(OBJDIR)/JacobiTests.o $(OBJDIR)/Jacobi.o
	$(CC) $(CCFLAGS) $(OBJDIR)/JacobiTests.o $(OBJDIR)/Jacobi.o -o cpuTests

$(OBJDIR)/JacobiTests.o: $(SRCDIR)/Jacobi.h $(TESTDIR)/JacobiTests.c
	$(CC) $(CCFLAGS) -c $(TESTDIR)/JacobiTests.c
	mv JacobiTests.o $(OBJDIR)/

rwdTests: $(OBJDIR)/JacobiRWDTests.o $(OBJDIR)/JacobiRWD.o
	$(GC) $(GCFLAGS) $(OBJDIR)/JacobiRWDTests.o $(OBJDIR)/JacobiRWD.o -o rwdTests

$(OBJDIR)/JacobiRWDTests.o: $(SRCDIR)/JacobiRWD.h $(TESTDIR)/JacobiRWDTests.cu
	$(GC) $(GCFLAGS) -c $(TESTDIR)/JacobiRWDTests.cu
	mv JacobiRWDTests.o $(OBJDIR)/

$(OBJDIR)/JacobiRWD.o: $(SRCDIR)/JacobiRWD.h $(SRCDIR)/JacobiRWD.cu
	$(GC) $(GCFLAGS) -c $(SRCDIR)/JacobiRWD.cu
	mv JacobiRWD.o $(OBJDIR)/

clean:
	rm -rf $(OBJDIR)/*.o jacobi rwdTests cpuTests
