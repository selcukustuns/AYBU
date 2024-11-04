CC = gcc
CFLAGS = -lm -lpthread
TARGET = hw1_20050111072
SOURCE = hw1_20050111072.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(SOURCE) -o $(TARGET) $(CFLAGS)

run: $(TARGET)
	./$(TARGET) input.jpg 4

clean:
	rm -f $(TARGET) *.jpg
