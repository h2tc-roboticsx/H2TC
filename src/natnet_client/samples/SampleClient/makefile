include = -I../../include
libpath = -L../../lib
libs = -lNatNetLibShared

all: build/SampleClient

build/SampleClient: 
	g++ SampleClient.cpp $(include) $(libpath) $(libs) -o build/SampleClient

.PHONY: clean
clean:
	@rm -f ./build/SampleClient