CXX = c++
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
BUILD_DIR ?= ../../outputs/cute/build
CXXFLAGS ?= -O0 -g -Wall -Wextra

ifndef CUTLASS_DIR
$(error Set CUTLASS_DIR to the CUTLASS source directory)
endif

CPPFLAGS += -isystem "$(CUTLASS_DIR)/include" -isystem "$(CUDA_HOME)/include"
SOURCES := $(wildcard t*.cpp)
PROGRAMS := $(patsubst %.cpp,$(BUILD_DIR)/%,$(SOURCES))

.PHONY: all check check-gpu check-wgmma-tma
all: $(PROGRAMS)

$(BUILD_DIR)/%: %.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 $< -o $@

$(BUILD_DIR)/sgemm_1: sgemm_1.cu | $(BUILD_DIR)
	"$(NVCC)" -std=c++17 -O3 -arch=$(CUDA_ARCH) -I"$(CUTLASS_DIR)/include" \
		-I"$(CUTLASS_DIR)/tools/util/include" $< -o $@

$(BUILD_DIR)/wgmma_tma_sm90: wgmma_tma_sm90.cu Makefile | $(BUILD_DIR)
	"$(NVCC)" -std=c++17 -O3 --expt-relaxed-constexpr \
		-gencode=arch=compute_90a,code=sm_90a -I"$(CUTLASS_DIR)/include" $< -o $@

$(BUILD_DIR):
	mkdir -p $@

check: all
	@set -e; for program in $(PROGRAMS); do "$$program"; done

check-gpu: $(BUILD_DIR)/sgemm_1
	"$<"

check-wgmma-tma: $(BUILD_DIR)/wgmma_tma_sm90
	"$<"
	"$<" 512 256 1024 T N
