# Daisy Seed / Hothouse — Programming Reference

## Quick Steps (every flash after the first)

1. **Enter DFU on the pedal** — hold both footswitches for 2 seconds; LEDs flash 3×, then the Daisy reboots into USB bootloader.
2. **Attach USB to WSL** — in Windows PowerShell (run as Admin or with usbipd rights):
   ```powershell
   usbipd list
   # Look for "STM32 Bootloader" — note its BUSID (e.g. 2-7)
   usbipd attach --wsl --busid 2-7
   ```
3. **Flash from WSL** — in the daisy build directory:
   ```bash
   cd results/pedalode/klonsim_v1/daisy
   make program-dfu
   ```
4. **Done** — the Daisy reboots and runs the new firmware automatically.

---

## First Flash (no firmware yet — use hardware buttons)

The DFU footswitch gesture requires firmware already running. For a blank/bricked Daisy:

1. With the Daisy **unplugged**, hold the **BOOT** button on the Seed.
2. Plug in USB (or tap **RESET** while holding BOOT). Release BOOT.
3. Attach to WSL (step 2 above) and flash (step 3 above).

---

## One-Time Setup

### Windows — install usbipd-win

Download and install from: https://github.com/dorssel/usbipd-win/releases

Verify:
```powershell
usbipd --version
```

### Bind the device (one time per machine, survives reboots)

```powershell
usbipd list
# Find "STM32 Bootloader", BUSID 2-7 (may change if you plug into a different port)
usbipd bind --busid 2-7
```

`bind` is persistent — you only need to re-run it if the BUSID changes (different physical USB port).

### WSL udev rule (one time per WSL instance)

Without this the flash fails with `LIBUSB_ERROR_ACCESS`:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/99-stm32-dfu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Build tools in WSL

```bash
sudo apt install make gcc-arm-none-eabi binutils-arm-none-eabi dfu-util
```

### Build libDaisy and DaisySP (one time)

```bash
cd /mnt/d/projects/libDaisy
git submodule update --init --recursive
make

cd /mnt/d/projects/DaisySP
make
```

---

## Building the Firmware

The firmware sources live in `results/pedalode/klonsim_v1/daisy/` and are generated
by the coefficient export script. To regenerate coefficients **and** copy sources:

```bash
# From repo root:
uv run python scripts/export_daisy_coeffs.py
```

Then build:

```bash
cd results/pedalode/klonsim_v1/daisy
make
```

Or regenerate + build in one shot via the Makefile target:

```bash
cd results/pedalode/klonsim_v1/daisy
make params && make
```

---

## Troubleshooting

### `usbipd attach` fails — device not in list

The Daisy only shows as `STM32 Bootloader (0483:df11)` while in DFU mode. If you see it as a generic audio device instead, it did not enter DFU.

### `No DFU capable USB device available`

The device is not visible to WSL. Check:
```bash
lsusb | grep 0483
```
If missing, re-run `usbipd attach --wsl --busid <id>` from PowerShell.

### `LIBUSB_ERROR_ACCESS`

udev rule is missing or hasn't reloaded yet. Re-run the udev commands above, then unplug/replug the Daisy.

### `Error during download get_status` (after successful flash)

This is **normal** and not an error. The Daisy reboots on the `:leave` command before
`dfu-util` can read the final status. The firmware was flashed successfully if you saw
the download progress reach 100%.

### BUSID changes

If you move the Daisy to a different USB port, the BUSID changes. Re-run:
```powershell
usbipd list          # find new BUSID
usbipd bind --busid <new-id>
usbipd attach --wsl --busid <new-id>
```

---

## Makefile Targets (in `results/pedalode/klonsim_v1/daisy/`)

| Target | What it does |
|---|---|
| `make` | Build ELF + BIN |
| `make program-dfu` | Flash via USB DFU (device must already be in DFU mode) |
| `make params` | Regenerate `klonsim_params.h` from `stage3_params.json` |
| `make clean` | Remove build artefacts |

---

## Hardware Notes

- **DFU gesture**: Hold **FOOTSWITCH_1 + FOOTSWITCH_2** simultaneously for 2 seconds. Both LEDs flash 3× then the Daisy resets to bootloader.
- **Effect toggle**: Tap **FOOTSWITCH_1** (left) — left LED on = effect active, off = bypass.
- **Heartbeat**: Move **TOGGLESWITCH_3** to UP — right LED blinks at 1 Hz to confirm firmware is running.
- **Sample rate**: 48 kHz (matches the ODE model training SR).
- **Signal chain**: `in → HPF1(80 Hz) → ×drive → tanh → HPF2(350 Hz) → Shelf(3 kHz, +2 dB) → ×level → out`
