CC := gcc
CFLAGS := -Wall -O0 -fopenmp

SRCS := $(wildcard *.c)
OBJS := $(SRCS:%.c=%.o)

.PHONY: all clean

all:: $(OBJS)

$(OBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
clean::
	rm -f *.o