// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/util/status_ring.h"

#include <chrono>
#include <future>
#include <tuple>
#include <utility>
#include <vector>

namespace spatialblend {

StatusRing::StatusRing(size_t ring_size)
    : ring_size_(ring_size), head_handle_(0) {}

int64_t StatusRing::Push(std::future<absl::Status> future) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Generate new event handle
  int64_t event_handle = next_event_handle_++;

  // If ring is full, advance head
  if (ring_.size() >= ring_size_) {
    ring_.pop_front();
    head_handle_++;
  }

  // Add new entry to ring
  ring_.emplace_back(event_handle, std::move(future), absl::Status());
  return event_handle;
}

absl::Status StatusRing::Get(int64_t event_handle, int timeout_ms) {
  if (!IsValidHandle(event_handle)) {
    return absl::InvalidArgumentError("Invalid event handle");
  }

  int64_t index = GetIndex(event_handle);
  if (index >= static_cast<int64_t>(ring_.size()) || index < 0) {
    return absl::NotFoundError("Event handle not found in ring");
  }

  auto& entry = ring_[index];
  if (std::get<0>(entry) != event_handle) {
    return absl::NotFoundError("Event handle not found in ring");
  }

  auto& future = std::get<1>(entry);
  auto& cached_status = std::get<2>(entry);

  // If status was already retrieved, return cached result
  if (future.valid() == false) {
    return cached_status;
  }

  // Wait for future with timeout
  if (future.wait_for(std::chrono::milliseconds(timeout_ms)) ==
      std::future_status::timeout) {
    return absl::DeadlineExceededError("Timeout waiting for status");
  }

  // Get and cache the status
  cached_status = future.get();
  return cached_status;
}

std::vector<std::pair<int64_t, absl::Status>> StatusRing::PollReady(
    size_t max_events) {
  std::vector<std::pair<int64_t, absl::Status>> ready;
  ready.reserve(max_events);

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : ring_) {
    if (ready.size() >= max_events) break;

    const int64_t handle = std::get<0>(entry);
    auto& future = std::get<1>(entry);
    auto& cached_status = std::get<2>(entry);

    if (!future.valid()) {
      continue;  // already consumed/cached
    }

    if (future.wait_for(std::chrono::milliseconds(0)) ==
        std::future_status::ready) {
      cached_status = future.get();  // consumes
      ready.emplace_back(handle, cached_status);
    }
  }
  return ready;
}

bool StatusRing::IsValidHandle(int64_t event_handle) const {
  return event_handle >= head_handle_ && event_handle < next_event_handle_;
}

int64_t StatusRing::GetIndex(int64_t event_handle) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!IsValidHandle(event_handle)) {
    return -1;
  }

  if (event_handle < head_handle_ || event_handle >= next_event_handle_) {
    return ring_.size();
  }

  int64_t relative_index = event_handle - head_handle_;
  if (relative_index >= static_cast<int64_t>(ring_.size())) {
    return ring_.size();
  }
  return relative_index;
}

}  // namespace spatialblend
