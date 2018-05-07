CC = g++ -std=c++11 -O3
CFLAGS = -g -Wall
SRCS = main.cpp

PROG = GPU_TEST

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

