CXX = c++
CUDA_HOME ?= /usr/local/cuda
BUILD_DIR ?= ../../outputs/cute/build
CXXFLAGS ?= -O0 -g -Wall -Wextra

ifndef CUTLASS_DIR
$(error Set CUTLASS_DIR to the CUTLASS source directory)
endif

CPPFLAGS += -isystem "$(CUTLASS_DIR)/include" -isystem "$(CUDA_HOME)/include"
SOURCES := $(wildcard t*.cpp)
PROGRAMS := $(patsubst %.cpp,$(BUILD_DIR)/%,$(SOURCES))

.PHONY: all check
all: $(PROGRAMS)

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $< -o $@

$(BUILD_DIR):
	mkdir -p $@

check: all
	@set -e; for program in $(PROGRAMS); do "$$program"; done
