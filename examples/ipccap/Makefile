CC=g++
LDFLAGS=-lpthread -lrt
CFLAGS=
TARGET=ipccap


ipccap.o : ipccap.cpp
	$(CC) -c ipccap.cpp $(CFLAGS)
op.o : op.cpp
	$(CC) -c op.cpp op.h $(CFLAGS)


OBJS=ipccap.o op.o
LIB=libop.a

all: $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LIB) $(LDFLAGS)

clean:
	$(RM) $(TARGET) $(OBJS)
