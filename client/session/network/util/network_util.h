// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_NETWORK_UTIL_NETWORK_UTIL_H_
#define SESSION_NETWORK_UTIL_NETWORK_UTIL_H_

#include <string>

#ifdef __ANDROID__
#include "session/network/util/android_network.h"
#endif

namespace spatialblend {
namespace network {
namespace util {

// Get the device's primary IPv4 address
// Returns "127.0.0.1" on Android if JNI is not available
std::string GetIPv4Address();

// Validate if a port number is valid (0-65535)
bool IsValidPort(int port);

// Validate if a host address is valid and reachable
// On Android, uses simplified validation to avoid permission issues
bool IsValidHostAddress(const std::string& host_address);

// Check if a port is available for binding
// On Android, uses simplified checking for ephemeral ports
bool IsPortAvailable(int port);

}  // namespace util
}  // namespace network
}  // namespace spatialblend

#endif  // SESSION_NETWORK_UTIL_NETWORK_UTIL_H_
