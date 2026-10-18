# KlonSim ODE model — Daisy Seed / Hothouse Makefile
#
# One-time setup (run once before first build):
#   make -C ../external/libDaisy && make -C ../external/DaisySP   (from daisy/, in WSL)
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

# Repo root; works from daisy/ and from the build copy in results/pedalode/klonsim_v1/daisy/.
PEDALDSP_DIR ?= $(shell git rev-parse --show-toplevel)

# Libraries are git submodules of this repo (external/). The D:\Projects clones are no longer used.
LIBDAISY_DIR  ?= $(PEDALDSP_DIR)/external/libDaisy
DAISYSP_DIR   ?= $(PEDALDSP_DIR)/external/DaisySP
HOTHOUSE_DIR  ?= $(PEDALDSP_DIR)/external/HothouseExamples

CPP_SOURCES = klonsim_pedal.cpp $(HOTHOUSE_DIR)/src/hothouse.cpp
C_INCLUDES  = -I$(HOTHOUSE_DIR)/src

SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile

# Regenerate params header + copy firmware sources, then build

.PHONY: params
params:
	# Python lives on Windows (.venv); WSL can run the Windows interpreter directly.
	cd $(PEDALDSP_DIR) && ./.venv/Scripts/python.exe scripts/export_daisy_coeffs.py
