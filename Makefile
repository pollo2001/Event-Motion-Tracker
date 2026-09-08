CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
INCLUDES = -I/usr/include
LIBS = -lOpenCL $(shell pkg-config --libs opencv4)

TARGET = event_tracker
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
