// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_C_SPATIALBLEND_SESSION_TYPE_H_
#define SESSION_C_SPATIALBLEND_SESSION_TYPE_H_
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t sb_handle_t;
typedef int64_t sb_inference_session_t;

/**
 * Types of inference models available.
 */
typedef enum {
  SB_INFERENCE_TYPE_GENERIC    = 0,  // Reserved
  SB_INFERENCE_TYPE_SAM3D      = 1,  // Legacy full pipeline (image+points -> glb+json)
  SB_INFERENCE_TYPE_SAM3_SEG   = 2,  // Segment only (image+points -> mask)
  SB_INFERENCE_TYPE_SAM3D_OBJ  = 3,  // Reconstruct only (image+mask -> glb+json)
} sb_inference_type_t;

/**
 * Input parameters for a generic inference request.
 */
typedef struct {
  const void* data;
  uint64_t data_size;
  const float* params;      // Model-specific (SAM3D: xyz triplets)
  uint64_t num_params;
  const char* output_path;
} sb_inference_input_t;

/**
 * 3D point coordinates.
 */
typedef struct {
  double x, y, z;
} sb_point_3d_t;

/**
 * Log severity levels.
 */
typedef enum {
  SB_LOG_SEVERITY_INFO = 0,
  SB_LOG_SEVERITY_WARNING = 1,
  SB_LOG_SEVERITY_ERROR = 2,
} sb_log_severity_t;

#ifdef __cplusplus
}
#endif

#endif /* SESSION_C_SPATIALBLEND_SESSION_TYPE_H_ */
