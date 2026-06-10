// maestro_render — the C ABI render entry (native-audio-rebuild spec §3.6).
//
// Called from the OS audio callback (AVAudioSourceNode render block) on the
// audio thread ONLY. `core` is the value of MaestroEngine.corePointer();
// `left`/`right` are non-interleaved float32 buffers of `frames` samples.
//
// Lifetime contract: the engine object owning `core` must outlive every call;
// stop the AVAudioEngine before releasing the MaestroEngine reference.

#ifndef MAESTRO_RENDER_H
#define MAESTRO_RENDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void maestro_render(void *core, float *left, float *right, uint32_t frames);

// Plan B (spec §3.3): pop frames from the render-pump ring. `pump` is the
// value of MaestroEngine.pumpPointer() after startPump(). On ring starvation
// the remainder is silence and an underrun is counted (snapshot-visible);
// the clock does not advance for unrendered frames — glitch, never drift.
void maestro_pump_consume(void *pump, float *left, float *right, uint32_t frames);

// Drift-injection sabotage selector for §6.4 negative-control runs.
// Always linkable; a no-op unless the library was built with the
// `drift-injection` feature (sabotaged validation builds only).
void maestro_set_sabotage(uint8_t mode);

#ifdef __cplusplus
}
#endif

#endif /* MAESTRO_RENDER_H */
