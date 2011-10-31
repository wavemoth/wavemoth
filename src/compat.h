#ifndef _WAVEMOTH_COMPAT_H_
#define _WAVEMOTH_COMPAT_H_

#if defined(_MSC_VER)
#define SSE_ALIGNED __declspec(align(16))
#else
#define SSE_ALIGNED __attribute__((aligned(16)))
#endif

#endif
