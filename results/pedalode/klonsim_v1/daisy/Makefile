# KlonSim ODE model — Daisy Seed / Hothouse Makefile
#
# One-time setup (run once before first build):
#   cd /mnt/d/projects/libDaisy && make
#
# Build (from results/pedalode/klonsim_v1/daisy/):
#   make
#
# Flash via USB DFU:
#   Hold BOOT on Seed, tap RESET, release BOOT, then:
#   make program-dfu
#
#   Or after the first flash, hold both Hothouse footswitches for 2 s
#   to reset to DFU, then: make program-dfu

TARGET = klonsim_pedal

# Standalone clones under D:\Projects
LIBDAISY_DIR  ?= /mnt/d/projects/libDaisy
DAISYSP_DIR   ?= /mnt/d/projects/DaisySP
HOTHOUSE_DIR  ?= /mnt/d/projects/HothouseExamples

CPP_SOURCES = klonsim_pedal.cpp $(HOTHOUSE_DIR)/src/hothouse.cpp
C_INCLUDES  = -I$(HOTHOUSE_DIR)/src

SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Regenerate params header + copy firmware sources, then build
PEDALDSP_DIR ?= $(shell git rev-parse --show-toplevel)

.PHONY: params
params:
	cd $(PEDALDSP_DIR) && uv run python scripts/export_daisy_coeffs.py
