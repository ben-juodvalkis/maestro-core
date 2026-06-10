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

#ifdef __cplusplus
}
#endif

#endif /* MAESTRO_RENDER_H */
