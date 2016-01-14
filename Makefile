CC_FLAGS := -arch=sm_30 -g -G

SRC = $(shell find . -name *.cu)

OBJ = $(SRC:%.cu=%.o)

BIN = lisp

all: $(BIN)

$(BIN): $(OBJ)
	nvcc $(CC_FLAGS) $(OBJ) -o lisp
%.o: %.cu
	nvcc -x cu $(CC_FLAGS) -Iinclude -dc $< -o $@

clean:
	rm -f src/*.o lisp

check: all
	./lisp test/test.lsp
