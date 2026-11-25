// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_UTIL_STATUS_RING_H_
#define SESSION_UTIL_STATUS_RING_H_

#include <deque>
#include <future>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace spatialblend {

typedef std::tuple<int64_t, std::future<absl::Status>, absl::Status> EventEntry;

// StatusRing is a ring buffer for storing statuses.
// - Push: Push a future to the ring.
// - Get: Get the status from the ring.
class StatusRing {
 public:
  StatusRing(size_t ring_size);
  ~StatusRing() = default;

  int64_t Push(std::future<absl::Status> future);
  absl::Status Get(int64_t event_handle, int timeout_ms);
  // Non-blocking: returns up to max_events completed handles + statuses.
  // Completed futures are consumed and cached (like Get()).
  std::vector<std::pair<int64_t, absl::Status>> PollReady(size_t max_events);
  bool IsValidHandle(int64_t event_handle) const;
  int64_t GetIndex(int64_t event_handle) const;

 private:
  mutable std::mutex mutex_;
  int64_t next_event_handle_ = 0;

  std::deque<EventEntry> ring_;
  const size_t ring_size_;
  int64_t head_handle_ = 0;
};

}  // namespace spatialblend

#endif  // SESSION_UTIL_STATUS_RING_H_
