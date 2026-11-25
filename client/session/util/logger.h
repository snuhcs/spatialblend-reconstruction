// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_UTIL_LOGGER_H_
#define SESSION_UTIL_LOGGER_H_

#include <functional>
#include <iostream>
#include <map>
#include <sstream>

#include "absl/status/status.h"
#include "session/common/common.h"

// create logger. cout by default
namespace spatialblend {

enum class LogSeverity {
  kInfo,
  kWarning,
  kError,
};

const char* ToString(LogSeverity severity);

class Logger {
 public:
  static Logger& Get();

  LogSeverity GetVerbosity() const;
  void SetVerbosity(LogSeverity severity);
  callback_handle_t SetReporter(
      LogSeverity verbosity,
      std::function<void(LogSeverity, const char*)> reporter);
  absl::Status RemoveReporter(callback_handle_t callback_id);
  std::pair<LogSeverity, std::string> GetLastLog() const;

  // DebugLog is only enabled in debug mode.
  void DebugLog(const char* format, ...);
  void Log(LogSeverity severity, const char* format, ...);

 private:
  void LogFormatted(LogSeverity severity, const char* format, va_list args);

  Logger() = default;
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  callback_handle_t next_callback_id_ = 0;
  std::map<
      callback_handle_t,
      std::pair<LogSeverity, std::function<void(LogSeverity, const char*)>>>
      reporters_;

  LogSeverity verbosity_ = LogSeverity::kInfo;
  std::pair<LogSeverity, std::string> last_message_;
};

}  // namespace spatialblend

#ifdef NDEBUG
#define SPATIALBLEND_LOG_DEBUG(format, ...) \
  do {                                   \
  } while (false);
#else
#define SPATIALBLEND_LOG_DEBUG(format, ...) \
  spatialblend::Logger::Get().DebugLog(format, ##__VA_ARGS__);
#endif

#define SPATIALBLEND_LOG(severity, format, ...) \
  spatialblend::Logger::Get().Log(severity, format, ##__VA_ARGS__);

#define SPATIALBLEND_NOT_IMPLEMENTED                                       \
  SPATIALBLEND_LOG(LogSeverity::kWarning, "Not implemented: %s", __FILE__, \
                __LINE__);

// Convenience macro for logging a statement *once* for a given process lifetime
#define SPATIALBLEND_LOG_ONCE(severity, format, ...)    \
  do {                                               \
    static const bool s_logged = [&] {               \
      SPATIALBLEND_LOG(severity, format, ##__VA_ARGS__) \
      return true;                                   \
    }();                                             \
    (void)s_logged;                                  \
  } while (false);

#define LOG_IF_ERROR(status)                                                  \
  if (!status.ok()) {                                                         \
    std::string err = std::string(status.message());                          \
    /* Only log if it's not an empty message and not a common cleanup failure \
     */                                                                       \
    if (!err.empty() && status.code() != absl::StatusCode::kUnavailable &&    \
        status.code() != absl::StatusCode::kNotFound) {                       \
      SPATIALBLEND_LOG(LogSeverity::kError, "Error: %s", err.c_str());           \
    }                                                                         \
  }

#define LOG_IF_ERROR_DEBUG(status)                                            \
  if (!status.ok()) {                                                         \
    std::string err = std::string(status.message());                          \
    /* Only log if it's not an empty message and not a common cleanup failure \
     */                                                                       \
    if (!err.empty() && status.code() != absl::StatusCode::kUnavailable &&    \
        status.code() != absl::StatusCode::kNotFound) {                       \
      SPATIALBLEND_LOG_DEBUG("Error: %s", err.c_str());                          \
    }                                                                         \
  }

#endif /* SESSION_UTIL_LOGGER_H_ */
