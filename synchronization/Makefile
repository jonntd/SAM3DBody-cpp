CC      := gcc
CFLAGS  := -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS := -lX11 -lXrandr -lqrencode -lm

TARGET  := xqr_time_sync
SRC     := libQRSync.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --ms --hz 30 --text --payload "t=%llu"

clean:
	rm -f $(TARGET)

