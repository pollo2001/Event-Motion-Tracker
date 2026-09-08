CXX = g++
CXXFLAGS = -O3 -Wall -std=c++17

# OpenCL and OpenCV dependencies
INCLUDES = -I/usr/include
LIBS = -lOpenCL `pkg-config --cflags --libs opencv4`

TARGET = event_tracker
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)
