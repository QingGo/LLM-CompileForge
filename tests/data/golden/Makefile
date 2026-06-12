# Build golden dylibs for contract testing.
# These are hand-written C kernels compiled to .dylib — no compiler dependency.
#
# Usage: make -f tests/data/golden/Makefile
#
# Each .dylib exports _mlir_ciface_<name> symbols with the same calling
# convention as MLIR-compiled dylibs (sret + packed memref descriptors).

CC := cc
CFLAGS := -O0 -fPIC -Wall -Wextra -Wno-unused-parameter

GOLDEN_DIR := tests/data/golden

all: $(GOLDEN_DIR)/matmul.dylib $(GOLDEN_DIR)/layer_norm.dylib $(GOLDEN_DIR)/sret_write.dylib $(GOLDEN_DIR)/rms_norm.dylib $(GOLDEN_DIR)/layer_norm_multi.dylib

$(GOLDEN_DIR)/sret_write.dylib: $(GOLDEN_DIR)/sret_write.c
	$(CC) $(CFLAGS) -shared -o $@ $<

contract: $(GOLDEN_DIR)/sret_layout_check.c
	$(CC) -c $< -o /dev/null && echo "✓ MemRef layout contract OK"

$(GOLDEN_DIR)/matmul.dylib: $(GOLDEN_DIR)/matmul.c
	$(CC) $(CFLAGS) -shared -o $@ $<

$(GOLDEN_DIR)/layer_norm.dylib: $(GOLDEN_DIR)/layer_norm.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

$(GOLDEN_DIR)/rms_norm.dylib: $(GOLDEN_DIR)/rms_norm.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

$(GOLDEN_DIR)/layer_norm_multi.dylib: $(GOLDEN_DIR)/layer_norm_multi.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

clean:
	rm -f $(GOLDEN_DIR)/matmul.dylib $(GOLDEN_DIR)/layer_norm.dylib $(GOLDEN_DIR)/rms_norm.dylib $(GOLDEN_DIR)/layer_norm_multi.dylib $(GOLDEN_DIR)/sret_write.dylib

.PHONY: all clean
