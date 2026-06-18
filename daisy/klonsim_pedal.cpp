// Hothouse multi-effect demo
//
// TOGGLESWITCH_1  UP     → Tremolo (5 Hz, 70% depth)
// TOGGLESWITCH_1  MIDDLE → KlonSim ODE distortion
// TOGGLESWITCH_1  DOWN   → Chorus  (0.5 Hz LFO, 20 ms center delay)
//
// FOOTSWITCH_1            → toggle bypass  (LED_1 on = engaged)
//
// TOGGLESWITCH_3  UP     → LED_2 blinks 1 Hz
// TOGGLESWITCH_3  MIDDLE → LED_2 off
// TOGGLESWITCH_3  DOWN   → LED_2 plays Morse "Brian"  (-... .-. .. .- -.)
//
// DFU reflash: hold both footswitches for 2 s, then  make program-dfu

#include "hothouse.h"
#include "klonsim_params.h"

#include <cmath>
#include <cstring>

using clevelandmusicco::Hothouse;
using daisy::AudioHandle;
using daisy::Led;
using daisy::SaiHandle;
using daisy::System;

static constexpr float SR = 48000.0f;

// ─── Direct Form II Transposed biquad ────────────────────────────────────────

struct Biquad {
    const float* b;
    const float* a;
    float w[2] = {0.0f, 0.0f};

    float Process(float x) noexcept {
        const float y = b[0] * x + w[0];
        w[0] = b[1] * x - a[0] * y + w[1];
        w[1] = b[2] * x - a[1] * y;
        return y;
    }
    void Reset() noexcept { w[0] = w[1] = 0.0f; }
};

// ─── Tremolo ──────────────────────────────────────────────────────────────────

struct Tremolo {
    float phase = 0.0f;
    static constexpr float kRateHz = 5.0f;
    static constexpr float kDepth  = 0.70f;  // 0 = no mod, 1 = full cut

    void Reset() noexcept { phase = 0.0f; }

    float Process(float x) noexcept {
        const float lfo = 0.5f * (1.0f + sinf(phase * 6.2831853f));
        phase += kRateHz / SR;
        if (phase >= 1.0f) phase -= 1.0f;
        return x * ((1.0f - kDepth) + kDepth * lfo);
    }
};

// ─── Chorus ───────────────────────────────────────────────────────────────────

struct Chorus {
    static constexpr int   kBufSize  = 2400;   // 50 ms at 48 kHz
    static constexpr float kRateHz   = 0.5f;
    static constexpr float kDepthMs  = 8.0f;
    static constexpr float kCenterMs = 20.0f;
    static constexpr float kMix      = 0.5f;

    float buf[kBufSize];
    int   write     = 0;
    float lfo_phase = 0.0f;

    void Reset() noexcept {
        memset(buf, 0, sizeof(buf));
        write = 0;
        lfo_phase = 0.0f;
    }

    float Process(float x) noexcept {
        const float lfo     = sinf(lfo_phase * 6.2831853f);
        lfo_phase += kRateHz / SR;
        if (lfo_phase >= 1.0f) lfo_phase -= 1.0f;

        const float delay_s = (kCenterMs + kDepthMs * lfo) * (SR / 1000.0f);
        const int   d_int   = static_cast<int>(delay_s);
        const float frac    = delay_s - d_int;
        const int   r1      = (write - d_int     + kBufSize) % kBufSize;
        const int   r2      = (write - d_int - 1 + kBufSize) % kBufSize;
        const float delayed = buf[r1] * (1.0f - frac) + buf[r2] * frac;

        buf[write] = x;
        write = (write + 1) % kBufSize;

        return x * (1.0f - kMix) + delayed * kMix;
    }
};

// ─── KlonSim ─────────────────────────────────────────────────────────────────

static Biquad klon_hpf1  = { klonsim::kHpf1B,  klonsim::kHpf1A  };
static Biquad klon_hpf2  = { klonsim::kHpf2B,  klonsim::kHpf2A  };
static Biquad klon_shelf = { klonsim::kShelfB, klonsim::kShelfA };

static void klonsim_reset() noexcept {
    klon_hpf1.Reset(); klon_hpf2.Reset(); klon_shelf.Reset();
}

static float klonsim_process(float x) noexcept {
    x *= klonsim::kInputGain;
    x  = klon_hpf1.Process(x);
    x  = tanhf(x * klonsim::kDrive);
    x  = klon_hpf2.Process(x);
    x  = klon_shelf.Process(x);
    x *= klonsim::kOutputLevel * 0.5623f;  // −5 dB vs params default
    return x;
}

// ─── Morse code player — "Brian"  (-... .-. .. .- -.) ────────────────────────
// Unit = 150 ms · dot = 150 ms · dash = 450 ms · letter gap = 450 ms

struct MorseStep { bool on; uint16_t ms; };

static constexpr MorseStep kBrianMorse[] = {
    // B  -...
    {true, 450}, {false, 150},
    {true, 150}, {false, 150},
    {true, 150}, {false, 150},
    {true, 150}, {false, 450},
    // r  .-.
    {true, 150}, {false, 150},
    {true, 450}, {false, 150},
    {true, 150}, {false, 450},
    // i  ..
    {true, 150}, {false, 150},
    {true, 150}, {false, 450},
    // a  .-
    {true, 150}, {false, 150},
    {true, 450}, {false, 450},
    // n  -.
    {true, 450}, {false, 150},
    {true, 150}, {false, 2000},  // long pause before repeat
};
static constexpr size_t kMorseLen = sizeof(kBrianMorse) / sizeof(kBrianMorse[0]);

struct MorsePlayer {
    size_t   idx       = 0;
    uint32_t deadline  = 0;
    bool     led_state = false;

    void Reset() noexcept { idx = 0; deadline = 0; led_state = false; }

    bool Update(uint32_t now_ms) noexcept {
        if (now_ms >= deadline) {
            led_state = kBrianMorse[idx].on;
            deadline  = now_ms + kBrianMorse[idx].ms;
            idx       = (idx + 1) % kMorseLen;
        }
        return led_state;
    }
};

// ─── Globals ──────────────────────────────────────────────────────────────────

static Hothouse hw;
static Led      led_bypass;
static Led      led_right;

static bool     bypassed = true;
static Tremolo  tremolo;
static Chorus   chorus;
static MorsePlayer morse;

static Hothouse::ToggleswitchPosition last_sw1 = Hothouse::TOGGLESWITCH_MIDDLE;

// ─── Audio callback ───────────────────────────────────────────────────────────

static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t                    size)
{
    hw.ProcessAllControls();

    if (hw.switches[Hothouse::FOOTSWITCH_1].RisingEdge()) {
        bypassed = !bypassed;
        led_bypass.Set(bypassed ? 0.0f : 1.0f);
    }
    led_bypass.Update();

    const auto sw1 = hw.GetToggleswitchPosition(Hothouse::TOGGLESWITCH_1);
    if (sw1 != last_sw1) {
        tremolo.Reset();
        chorus.Reset();
        klonsim_reset();
        last_sw1 = sw1;
    }

    for (size_t i = 0; i < size; ++i) {
        float x = in[0][i];

        if (!bypassed) {
            switch (sw1) {
                case Hothouse::TOGGLESWITCH_UP:
                    x = tremolo.Process(x);
                    break;
                case Hothouse::TOGGLESWITCH_MIDDLE:
                    x = klonsim_process(x);
                    break;
                case Hothouse::TOGGLESWITCH_DOWN:
                    x = chorus.Process(x);
                    break;
            }
        }

        out[0][i] = out[1][i] = x;
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    hw.Init();
    hw.SetAudioBlockSize(4);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    led_bypass.Init(hw.seed.GetPin(Hothouse::LED_1), false);
    led_bypass.Set(0.0f);
    led_bypass.Update();

    led_right.Init(hw.seed.GetPin(Hothouse::LED_2), false);
    led_right.Set(0.0f);
    led_right.Update();

    chorus.Reset();

    hw.StartAdc();
    hw.StartAudio(AudioCallback);

    uint32_t last_blink_ms = 0;
    bool     blink_state   = false;

    while (true) {
        hw.DelayMs(10);
        hw.CheckResetToBootloader();

        const uint32_t now = System::GetNow();
        const auto     sw3 = hw.GetToggleswitchPosition(Hothouse::TOGGLESWITCH_3);

        float right_brightness = 0.0f;

        switch (sw3) {
            case Hothouse::TOGGLESWITCH_UP:
                if (now - last_blink_ms >= 500) {
                    blink_state   = !blink_state;
                    last_blink_ms = now;
                }
                right_brightness = blink_state ? 1.0f : 0.0f;
                break;

            case Hothouse::TOGGLESWITCH_MIDDLE:
                right_brightness = 0.0f;
                blink_state      = false;
                last_blink_ms    = now;
                morse.Reset();
                break;

            case Hothouse::TOGGLESWITCH_DOWN:
                right_brightness = morse.Update(now) ? 1.0f : 0.0f;
                blink_state      = false;
                last_blink_ms    = now;
                break;
        }

        led_right.Set(right_brightness);
        led_right.Update();
    }
}
