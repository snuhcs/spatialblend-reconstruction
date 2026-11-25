// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/network/util/network_util.h"

#ifdef __ANDROID__
#include "session/network/util/android_network.h"
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// Use standard ifaddrs.h - Android NDK provides this since API 21
#include <ifaddrs.h>
#endif

#include <cmath>
#include <cstring>
#include <iostream>

#include "session/util/logger.h"

namespace spatialblend {
namespace network {
namespace util {

std::string GetIPv4Address() {
#ifdef SPATIALBLEND_TEST
  return "localhost";
#endif

#ifdef __ANDROID__
  // On Android, use JNI ConnectivityManager APIs instead of getifaddrs
  // This avoids SELinux permission denials that occur with getifaddrs

  // Try to auto-initialize if not already initialized
  if (!AndroidNetworkUtils::IsInitialized()) {

    // Try to get JNI environment and auto-detect context
    JNIEnv* env = AndroidNetworkUtils::GetJNIEnv();
    if (env) {
      // Try to initialize with auto-detected context (no Java-side
      // initialization needed)
      AndroidNetworkUtils::InitializeWithAutoContext(env);
    }
  }

  return AndroidNetworkUtils::GetDeviceIPAddress();
#elif defined(_WIN32)
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return "";
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    WSACleanup();
    return "";
  }

  struct addrinfo hints = {}, *result = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(hostname, nullptr, &hints, &result) != 0) {
    WSACleanup();
    return "";
  }

  char ipstr[INET_ADDRSTRLEN];
  for (struct addrinfo* ptr = result; ptr != nullptr; ptr = ptr->ai_next) {
    struct sockaddr_in* addr = (struct sockaddr_in*)ptr->ai_addr;
    inet_ntop(AF_INET, &(addr->sin_addr), ipstr, sizeof(ipstr));
    std::string ip_address(ipstr);
    if (ip_address != "127.0.0.1") {
      freeaddrinfo(result);
      WSACleanup();
      return ip_address;
    }
  }

  freeaddrinfo(result);
  WSACleanup();
  return "";
#else  // Linux/Unix (non-Android)
  struct ifaddrs *ifaddr, *ifa;
  int family;
  char host[NI_MAXHOST];

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return "";
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;

    family = ifa->ifa_addr->sa_family;
    if (family == AF_INET) {
      if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                      NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
        std::string ip_address(host);
        if (ip_address == "127.0.0.1") continue;
        freeifaddrs(ifaddr);
        return ip_address;
      }
    }
  }

  freeifaddrs(ifaddr);
  return "";
#endif
}

bool IsValidPort(int port) { return port >= 0 && port <= 65535; }

bool IsValidHostAddress(const std::string& host_address) {
#ifdef _WIN32
  // Windows-specific initialization
  WSADATA wsaData;
  bool needs_cleanup = false;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return false;
  }
  needs_cleanup = true;
#endif

  bool is_valid = false;
  do {
    // Check if the address is empty
    if (host_address.empty()) {
      break;
    }

    // Special case for localhost and loopback addresses
    if (host_address == "localhost" || host_address == "127.0.0.1") {
      is_valid = true;
      break;
    }

    // Calculate dot count for IP address validation
    size_t dot_count =
        std::count(host_address.begin(), host_address.end(), '.');

#ifdef __ANDROID__
    // On Android, be more permissive with address validation
    // to avoid network discovery that requires special permissions
    if (host_address == "127.0.0.1" || host_address == "localhost") {
      is_valid = true;
      break;
    }

    // For Android, also accept basic IP format validation without DNS lookup
    if (dot_count == 3) {
      struct sockaddr_in sa;
      int result = inet_pton(AF_INET, host_address.c_str(), &(sa.sin_addr));
      if (result == 1) {
        is_valid = true;
        break;
      }
    }
#endif

    // For IP addresses, verify format
    if (dot_count == 3) {
      struct sockaddr_in sa;
      int result = inet_pton(AF_INET, host_address.c_str(), &(sa.sin_addr));
      if (result != 1) {
        break;
      }
    } else if (dot_count == 7) {
      struct sockaddr_in6 sa;
      int result = inet_pton(AF_INET6, host_address.c_str(), &(sa.sin6_addr));
      if (result != 1) {
        break;
      }
    } else if (dot_count != 1) {
      // else, it is an invalid host address unless it is a domain name
      break;
    }

    // Try to resolve the address
    struct addrinfo hints = {}, *result = nullptr;
    hints.ai_family = AF_UNSPEC;      // Allow IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP socket

    // Use getaddrinfo to validate the address
    int status = getaddrinfo(host_address.c_str(), nullptr, &hints, &result);

    if (status == 0 && result != nullptr) {
      is_valid = true;
      freeaddrinfo(result);
    }
  } while (false);

#ifdef _WIN32
  if (needs_cleanup) {
    WSACleanup();
  }
#endif

  return is_valid;
}

bool IsPortAvailable(int port) {
  // First check if the port number is valid
  if (!IsValidPort(port)) {
    return false;
  }

#ifdef __ANDROID__
  // On Android, avoid socket operations that might trigger SELinux denials
  // Just return true for valid ports in the ephemeral range
  // This is a simplified approach for Android sandboxed environment
  // TODO: better port availability check for grpc_server_base port already exists test
  return port >= 49152 && port <= 65535;
#endif

#ifdef _WIN32
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return false;
  }

  SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == INVALID_SOCKET) {
    WSACleanup();
    return false;
  }

  BOOL opt = TRUE;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) !=
      0) {
    closesocket(sock);
    WSACleanup();
    return false;
  }

  sockaddr_in addr;
  ZeroMemory(&addr, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<u_short>(port));

  bool is_available = (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);

  closesocket(sock);
  WSACleanup();
  return is_available;
#else
  // Create a socket
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    return false;  // Failed to create socket
  }

  // Set socket option to allow reusing the address (helpful for testing)
  int opt = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    close(sock);
    return false;
  }

  // Set up the address structure
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  // Try to bind to the port
  bool is_available = (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);

  // Clean up
  close(sock);

  return is_available;
#endif
}

}  // namespace util
}  // namespace network
}  // namespace spatialblend
