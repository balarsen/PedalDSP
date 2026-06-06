"""Utility helpers for synthesis, analysis, and visualisation."""
from .synthesis import guitar_note, g_major_chord, white_noise_id, pure_sine
from .analysis import (
    safe_trim,
    compute_esr_skip,
    harmonic_frequencies,
    apply_effect,
    peak_db,
    rms_db,
)
from .plotting import (
    db_spectrum,
    freq_response_db,
    filter_impulse_response,
    mark_harmonics,
    mark_vlines,
    plot_waveform,
    plot_spectrum,
    plot_impulse_response,
    plot_freq_response,
    compare_waveforms,
    compare_spectra,
    signal_dashboard,
    error_analysis_panel,
    ir_overlay,
    freq_response_overlay,
    model_fit_panel,
)
