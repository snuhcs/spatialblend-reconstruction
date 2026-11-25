// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_COMMON_COMMON_H_
#define SESSION_COMMON_COMMON_H_

#include <cstdint>
#include <functional>

#include "absl/status/status.h"

namespace spatialblend {

typedef int64_t callback_handle_t;

const callback_handle_t kInvalidCallbackHandle = -1;

}  // namespace spatialblend

#endif /* SESSION_COMMON_COMMON_H_ */
