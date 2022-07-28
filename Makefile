CC:=clang++
FLAGS:=-Wall -Wpedantic -std=c++17 -g3 -Iinclude
CUDA_FLAGS:=--cuda-gpu-arch=sm_86 -Xclang -fcuda-allow-variadic-functions -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -lcudart -ldl -lrt -lpthread
RELEASE_FLAGS:=-O3 -DNDEBUG -ffast-math

HEADERS:=$(wildcard include/*.h include/*.cuh include/*.hpp)
CXX_SOURCES:=$(wildcard src/*.cpp)
SOURCES:=$(wildcard src/*.cu)
CXX_TESTS:=$(wildcard test/*.cpp)
TESTS:=$(wildcard test/*.cu)

PREFIX=./env

.PHONY: all tests compdb clean install

all: compdb tests

release: FLAGS += ${RELEASE_FLAGS}

release: tests

tests: ${TESTS:test/%.cu=build/%} ${HEADERS}

compdb: Makefile
	compiledb --no-build make

clean:
	rm -f build/*

install: ${HEADERS}
	rm -rf ${PREFIX}/include/SimpleMatrix
	mkdir ${PREFIX}/include/SimpleMatrix
	cp include/* ${PREFIX}/include/SimpleMatrix

${TESTS:test/%.cu=build/%}: build/%: test/%.cu ${SOURCES:src/%.cu=build/%.o} ${CXX_SOURCES:src/%.cpp=build/%.o}
	${CC} ${FLAGS} ${CUDA_FLAGS} -o $@ $^

${CXX_SOURCES:src/%.cpp=build/%.o}: build/%.o: %.cpp ${HEADERS}
	${CC} ${FLAGS} -c -o $@ $<

${SOURCES:src/%.cu=build/%.o}: build/%.o: %.cu ${HEADERS}
	${CC} ${FLAGS} ${CUDA_FLAGS} -c -o $@ $<
