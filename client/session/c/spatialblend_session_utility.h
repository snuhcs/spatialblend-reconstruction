// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_C_SPATIALBLEND_SESSION_UTILITY_H_
#define SESSION_C_SPATIALBLEND_SESSION_UTILITY_H_

#include <stdint.h>

#include "session/c/spatialblend_session_type.h"

#if defined(_WIN32)
#if defined(SPATIALBLEND_BUILD_DLL)
#define SPATIALBLEND_EXPORT __declspec(dllexport)
#else
#define SPATIALBLEND_EXPORT __declspec(dllimport)
#endif
#else
#define SPATIALBLEND_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wait for an event with a timeout.
 * @param event_handle The event handle to wait for
 * @param timeout_ms Timeout in milliseconds
 * @return true if event was signaled, false if timeout occurred
 */
SPATIALBLEND_EXPORT bool spatialblend_try_wait_for_event(sb_handle_t event_handle,
                                                   int64_t timeout_ms);

/**
 * Get an error message from a failed event.
 * @param event_handle The event handle that failed
 * @return A string containing the error message.
 */
SPATIALBLEND_EXPORT const char* spatialblend_get_error_message(
    sb_handle_t event_handle);

#ifdef __cplusplus
}
#endif

#endif /* SESSION_C_SPATIALBLEND_SESSION_UTILITY_H_ */
