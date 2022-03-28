CC=gcc
CFLAGS=-O3 -march=native -Wall

OBJS=cpucycles.o decrypt.o encode.o encrypt.o extract.o fastrandombytes.o fft.o keygen.o mat.o ntt.o poly.o randombytes.o sample_fft.o sample_z.o test.o
PROGRAM=latte

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) $(OBJS) -o $(PROGRAM) -lgmp -lmpfr -lmpc -lm -lXKCP

clean:;
	rm -f *.o $(PROGRAM)

