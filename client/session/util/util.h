// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_UTIL_UTIL_H_
#define SESSION_UTIL_UTIL_H_

#include <future>
#include <string>

#include "session/common/common.h"

namespace spatialblend {
namespace util {

// Helper function to create a future that is already ready.
template <typename T>
std::future<T> MakeReadyFuture(T value) {
  std::promise<T> promise;
  promise.set_value(value);
  return promise.get_future();
}

template <typename T>
bool IsFutureReady(std::future<T>& future) {
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

}  // namespace util
}  // namespace spatialblend

#define RETURN_IF_ERROR(status) \
  if (!status.ok()) {           \
    return status;              \
  }

#define RETURN_VALUE_IF_ERROR(status, return_value) \
  if (!status.ok()) {                               \
    return return_value;                            \
  }

#define RETURN_FUTURE_IF_ERROR(status) \
  RETURN_VALUE_IF_ERROR(status, util::MakeReadyFuture(status))

#define RUNTIME_ERROR_IF_ERROR(status)                       \
  if (!status.ok()) {                                        \
    throw std::runtime_error(std::string(status.message())); \
  }

#endif /* SESSION_UTIL_UTIL_H_ */
