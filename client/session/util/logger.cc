// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/util/logger.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>

#include "absl/strings/str_format.h"

namespace spatialblend {

const char* ToString(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::kInfo:
      return "INFO";
    case LogSeverity::kWarning:
      return "WARNING";
    case LogSeverity::kError:
      return "ERROR";
  }
  return "";
}

Logger& Logger::Get() {
  static Logger* logger = new Logger;
  return *logger;
}

LogSeverity Logger::GetVerbosity() const { return verbosity_; }

void Logger::SetVerbosity(LogSeverity severity) { verbosity_ = severity; }

callback_handle_t Logger::SetReporter(
    LogSeverity verbosity,
    std::function<void(LogSeverity, const char*)> reporter) {
  static std::mutex reporter_mutex;
  std::lock_guard<std::mutex> lock(reporter_mutex);
  callback_handle_t id = next_callback_id_++;
  reporters_[id] = {verbosity, reporter};
  return id;
}

absl::Status Logger::RemoveReporter(callback_handle_t callback_id) {
  static std::mutex reporter_mutex;
  std::lock_guard<std::mutex> lock(reporter_mutex);
  if (reporters_.find(callback_id) == reporters_.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given callback id does not exist. %d",
                        static_cast<int>(callback_id)));
  }
  reporters_.erase(callback_id);
  return absl::OkStatus();
}

std::pair<LogSeverity, std::string> Logger::GetLastLog() const {
  return last_message_;
}

void Logger::DebugLog(const char* format, ...) {
#ifndef NDEBUG
  va_list args;
  va_start(args, format);
  LogFormatted(LogSeverity::kInfo, format, args);
  va_end(args);
#endif
}

void Logger::Log(LogSeverity severity, const char* format, ...) {
  bool log_to_console = (verbosity_ <= severity);

  bool needs_formatting = false;
  {
    static std::mutex reporter_mutex;
    std::lock_guard<std::mutex> lock(reporter_mutex);
    for (const auto& pair : reporters_) {
      if (pair.second.first <= severity) {
        needs_formatting = true;
        break;
      }
    }
  }

  if (!log_to_console && !needs_formatting) {
    return;
  }

  std::string formatted_message;
  va_list args;
  va_start(args, format);
  va_list args_copy;
  va_copy(args_copy, args);
  int size = vsnprintf(nullptr, 0, format, args_copy);
  va_end(args_copy);

  if (size >= 0) {
    size++;
    std::vector<char> buffer(size);
    vsnprintf(buffer.data(), size, format, args);
    formatted_message = buffer.data();
  }
  va_end(args);

  if (log_to_console) {
    static std::mutex console_mutex;
    std::lock_guard<std::mutex> lock(console_mutex);
    fprintf(stderr, "%s: %s\n", ToString(severity), formatted_message.c_str());
  }

  {
    static std::mutex reporter_mutex;
    std::lock_guard<std::mutex> lock(reporter_mutex);
    for (const auto& pair : reporters_) {
      if (pair.second.first <= severity) {
        pair.second.second(severity, formatted_message.c_str());
      }
    }
  }

  last_message_ = std::make_pair(severity, formatted_message);
}

void Logger::LogFormatted(LogSeverity severity, const char* format,
                          va_list args) {}

}  // namespace spatialblend
