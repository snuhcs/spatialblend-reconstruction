// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "session/inference/inference_client.h"
#include "session/inference/sam3d/sam3d_client.h"

namespace spatialblend::sam3d {
namespace {

std::string GetEnvOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

void LogEnv(const char* name) {
  const std::string v = GetEnvOrEmpty(name);
  if (v.empty()) {
    GTEST_LOG_(INFO) << "env[" << name << "] = <empty>";
  } else {
    GTEST_LOG_(INFO) << "env[" << name << "] = " << v;
  }
}

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  const char last = dir.back();
  if (last == '/' || last == '\\') return dir + file;
  return dir + "/" + file;
}

std::string TimestampForFilename() {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const std::time_t t = system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return std::string(buf);
}

std::string GetStableOutputDirOrEmpty() {
  return GetEnvOrEmpty("SPATIALBLEND_SAM3D_TEST_OUTPUT_DIR");
}

std::string GetOutputsDir() {
  return GetEnvOrEmpty("TEST_UNDECLARED_OUTPUTS_DIR");
}

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::string s((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
  return s;
}

std::string ReadFileBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::string s((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
  return s;
}

std::string GetWorkspaceName() {
  std::string ws = GetEnvOrEmpty("TEST_WORKSPACE");
  return ws.empty() ? std::string("_main") : ws;
}

std::string GetRunfilesDir() {
  std::string runfiles_dir = GetEnvOrEmpty("RUNFILES_DIR");
  if (!runfiles_dir.empty()) return runfiles_dir;

  std::string test_srcdir = GetEnvOrEmpty("TEST_SRCDIR");
  if (!test_srcdir.empty()) return test_srcdir;

  std::string manifest = GetEnvOrEmpty("RUNFILES_MANIFEST_FILE");
  if (!manifest.empty()) {
    const std::string suffix = ".runfiles_manifest";
    if (manifest.size() > suffix.size() &&
        manifest.substr(manifest.size() - suffix.size()) == suffix) {
      return manifest.substr(0, manifest.size() - suffix.size()) + ".runfiles";
    }
  }
  return {};
}

std::string ResolveRunfilePath(const std::string& workspace_relative_path) {
  const std::string runfiles_dir = GetRunfilesDir();
  const std::string ws = GetWorkspaceName();
  if (runfiles_dir.empty()) {
    return workspace_relative_path;
  }
  return (std::filesystem::path(runfiles_dir) / ws / workspace_relative_path)
      .string();
}

std::vector<std::string> ListFilesByExt(const std::string& dir,
                                        const std::vector<std::string>& exts) {
  std::vector<std::string> out;
  std::error_code ec;
  const std::filesystem::path p(dir);
  if (!std::filesystem::exists(p, ec)) return out;
  for (const auto& entry : std::filesystem::directory_iterator(p, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const std::string ext = entry.path().extension().string();
    auto it = std::find_if(exts.begin(), exts.end(), [&](const std::string& e) {
      if (ext.size() != e.size()) return false;
      for (size_t i = 0; i < ext.size(); ++i) {
        const char a = static_cast<char>(std::tolower(ext[i]));
        const char b = static_cast<char>(std::tolower(e[i]));
        if (a != b) return false;
      }
      return true;
    });
    if (it != exts.end()) out.push_back(entry.path().string());
  }
  return out;
}

// Reads 2D points from file and returns flat float array [x1,y1,z1, x2,y2,z2, ...]
std::vector<float> ReadPoints2DFileAsFlat(const std::string& path) {
  std::ifstream in(path);
  std::vector<float> params;
  if (!in) return params;

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    double x = 0.0, y = 0.0;
    char comma = '\0';
    std::stringstream ss(line);
    ss >> x;
    ss >> comma;
    ss >> y;
    if (!ss.fail()) {
      params.push_back(static_cast<float>(x));
      params.push_back(static_cast<float>(y));
      params.push_back(0.0f);
    }
  }
  return params;
}

size_t PickRandomIndex(size_t n) {
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, n - 1);
  return dist(rng);
}

}  // namespace

TEST(Sam3dRemoteTest, MultiUserConcurrencyLikePythonRunner) {
  GTEST_LOG_(INFO) << "=== Sam3dRemoteTest starting ===";
  LogEnv("SPATIALBLEND_SAM3D_ADDR");
  LogEnv("SPATIALBLEND_SAM3D_TEST_OUTPUT_DIR");
  LogEnv("TEST_TMPDIR");
  LogEnv("TEST_UNDECLARED_OUTPUTS_DIR");
  LogEnv("TEST_SRCDIR");
  LogEnv("RUNFILES_DIR");
  LogEnv("RUNFILES_MANIFEST_FILE");
  LogEnv("TEST_WORKSPACE");

  const std::string addr = GetEnvOrEmpty("SPATIALBLEND_SAM3D_ADDR");
  if (addr.empty()) {
    FAIL() << "SPATIALBLEND_SAM3D_ADDR is not set. This test requires a running "
              "remote server (host:port) and should not be skipped.";
  }

  const std::string tmpdir = GetEnvOrEmpty("TEST_TMPDIR");
  const std::string outputs_dir = GetOutputsDir();
  const std::string out_dir =
      !outputs_dir.empty() ? outputs_dir : (tmpdir.empty() ? "." : tmpdir);
  GTEST_LOG_(INFO) << "Output directory: " << out_dir;

  const std::vector<std::string> users = {"Alice", "Bob", "Charlie", "Dave"};

  std::vector<int> delays_s;
  delays_s.reserve(users.size());
  {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(3, 13);
    for (size_t i = 0; i < users.size(); ++i) {
      delays_s.push_back(dist(rng));
    }
    std::sort(delays_s.begin(), delays_s.end());
  }
  {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < delays_s.size(); ++i) {
      if (i) oss << ", ";
      oss << delays_s[i];
    }
    oss << "]";
    GTEST_LOG_(INFO) << "Delays (seconds, sorted): " << oss.str();
  }

  std::mutex mu;
  std::condition_variable cv;
  int done = 0;

  struct Outcome {
    absl::Status status;
    std::string output_path;
  };
  std::vector<Outcome> outcomes(users.size());

  std::vector<std::thread> threads;
  threads.reserve(users.size());

  const std::string images_dir =
      ResolveRunfilePath("session/inference/sam3d/test_data/images");
  const std::string points_dir =
      ResolveRunfilePath("session/inference/sam3d/test_data/points");

  GTEST_LOG_(INFO) << "Resolved images_dir: " << images_dir;
  GTEST_LOG_(INFO) << "Resolved points_dir: " << points_dir;

  const std::vector<std::string> image_paths =
      ListFilesByExt(images_dir, {".png", ".jpg", ".jpeg", ".webp"});
  const std::vector<std::string> point_paths =
      ListFilesByExt(points_dir, {".txt"});

  GTEST_LOG_(INFO) << "Found images: " << image_paths.size();
  for (const auto& p : image_paths) {
    GTEST_LOG_(INFO) << "  image: " << p;
  }
  GTEST_LOG_(INFO) << "Found point files: " << point_paths.size();
  for (const auto& p : point_paths) {
    GTEST_LOG_(INFO) << "  points: " << p;
  }

  ASSERT_FALSE(image_paths.empty())
      << "No images found under: " << images_dir
      << " (check Bazel data/runfiles)";
  ASSERT_FALSE(point_paths.empty())
      << "No point files found under: " << points_dir
      << " (check Bazel data/runfiles)";

  std::atomic<int> rpc_attempts{0};

  for (size_t i = 0; i < users.size(); ++i) {
    outcomes[i].output_path = JoinPath(out_dir, "result_" + users[i] + ".txt");

    threads.emplace_back([&, i]() {
      std::this_thread::sleep_for(std::chrono::seconds(delays_s[i]));

      GTEST_LOG_(INFO) << "[" << users[i] << "] starting client thread";
      auto client = std::make_shared<Sam3dClient>();
      absl::Status st = client->ConnectToAddress(addr);
      if (!st.ok()) {
        GTEST_LOG_(INFO) << "[" << users[i] << "] ConnectToAddress failed: "
                         << st.ToString();
        std::lock_guard<std::mutex> lock(mu);
        outcomes[i].status = st;
        done++;
        cv.notify_all();
        return;
      }
      GTEST_LOG_(INFO) << "[" << users[i] << "] connected";

      const std::string image_path = image_paths[PickRandomIndex(image_paths.size())];
      const std::string stem = std::filesystem::path(image_path).stem().string();
      std::string points_path;
      for (const auto& p : point_paths) {
        if (std::filesystem::path(p).stem().string() == stem) {
          points_path = p;
          break;
        }
      }
      if (points_path.empty()) {
        std::lock_guard<std::mutex> lock(mu);
        outcomes[i].status = absl::InvalidArgumentError(
            "No matching points file for selected image basename");
        done++;
        cv.notify_all();
        return;
      }

      const std::string image_bytes = ReadFileBinary(image_path);
      std::vector<float> params = ReadPoints2DFileAsFlat(points_path);
      if (image_bytes.empty()) {
        GTEST_LOG_(INFO) << "[" << users[i] << "] Failed to read image file: "
                         << image_path;
        std::lock_guard<std::mutex> lock(mu);
        outcomes[i].status = absl::InternalError("Failed to read image file");
        done++;
        cv.notify_all();
        return;
      }
      if (params.empty()) {
        GTEST_LOG_(INFO) << "[" << users[i] << "] Failed to read points file: "
                         << points_path;
        std::lock_guard<std::mutex> lock(mu);
        outcomes[i].status = absl::InternalError("Failed to read points file");
        done++;
        cv.notify_all();
        return;
      }

      const size_t num_points = params.size() / 3;
      GTEST_LOG_(INFO) << "[" << users[i] << "] Selected image: " << image_path
                       << " (" << image_bytes.size() << " bytes)";
      GTEST_LOG_(INFO) << "[" << users[i] << "] Selected points: " << points_path
                       << " (" << num_points << " points)";

      rpc_attempts.fetch_add(1, std::memory_order_relaxed);
      GTEST_LOG_(INFO) << "[" << users[i] << "] Sending Process RPC -> " << addr;
      auto fut = client->Process(std::to_string(i), image_bytes,
                                 params.data(), params.size());
      absl::StatusOr<inference::InferenceResult> rpc = fut.get();
      absl::Status final_status = rpc.status();
      if (final_status.ok() && rpc->success == false) {
        final_status = absl::InternalError(rpc->message);
      }

      if (!final_status.ok()) {
        GTEST_LOG_(INFO) << "[" << users[i] << "] RPC failed: "
                         << final_status.ToString();
      } else {
        GTEST_LOG_(INFO) << "[" << users[i] << "] RPC success. message="
                         << rpc->message << "\n glb_bytes="
                         << rpc->output_data.size();
      }

      if (final_status.ok()) {
        const std::string base_path =
            JoinPath(out_dir, "result_" + users[i]);
        const std::string txt_path = base_path + ".txt";
        const std::string glb_path = base_path + "_objmodel.glb";
        const std::string json_path = base_path + "_objinfo.json";

        const std::string& glb_payload = rpc->output_data;
        const std::string& json_payload = rpc->json_data;

        // Write GLB
        if (final_status.ok()) {
          std::ofstream glb(glb_path, std::ios::binary);
          if (glb) {
            glb.write(glb_payload.data(),
                      static_cast<std::streamsize>(glb_payload.size()));
            glb.close();
          }
        }

        // Write JSON objinfo
        if (final_status.ok() && !json_payload.empty()) {
          std::ofstream jf(json_path, std::ios::binary);
          if (jf) {
            jf.write(json_payload.data(),
                     static_cast<std::streamsize>(json_payload.size()));
            jf.close();
          }
        }

        if (final_status.ok()) {
          std::ofstream out(txt_path);
          if (!out) {
            final_status = absl::InternalError("Failed to open output file");
          } else {
            out << "User: " << users[i] << "\n";
            out << "Success: " << (rpc->success ? "true" : "false") << "\n";
            out << "Message: " << rpc->message << "\n";
            out << "GLB_Bytes: " << glb_payload.size() << "\n";
            out << "JSON_Bytes: " << json_payload.size() << "\n";
            out.close();
            if (!out) {
              final_status = absl::InternalError("Failed to write output file");
            }
          }
        }

        if (final_status.ok()) {
          outcomes[i].output_path = txt_path;
          GTEST_LOG_(INFO) << "[" << users[i] << "] Wrote outputs:\n"
                           << "  txt:  " << txt_path << "\n"
                           << "  glb:  " << glb_path << "\n"
                           << "  json: " << json_path;

          const std::string stable_dir = GetStableOutputDirOrEmpty();
          if (!stable_dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(stable_dir, ec);
            if (ec) {
              final_status = absl::InternalError(
                  "Failed to create SPATIALBLEND_SAM3D_TEST_OUTPUT_DIR: " +
                  stable_dir + " err=" + ec.message());
            } else {
              const std::string ts = TimestampForFilename();
              const std::string stable_base =
                  JoinPath(stable_dir, "result_" + users[i] + "_" + ts);
              const std::string txt_copy = stable_base + ".txt";
              const std::string glb_copy = stable_base + "_objmodel.glb";
              const std::string json_copy = stable_base + "_objinfo.json";

              // Copy GLB
              {
                std::ofstream glb2(glb_copy, std::ios::binary);
                if (!glb2) {
                  final_status =
                      absl::InternalError("Failed to open stable glb output: " +
                                          glb_copy);
                } else {
                  glb2.write(glb_payload.data(),
                             static_cast<std::streamsize>(glb_payload.size()));
                  glb2.close();
                  if (!glb2) {
                    final_status = absl::InternalError(
                        "Failed to write stable glb output: " + glb_copy);
                  }
                }
              }

              // Copy JSON
              if (final_status.ok() && !json_payload.empty()) {
                std::ofstream jf2(json_copy, std::ios::binary);
                if (!jf2) {
                  final_status = absl::InternalError(
                      "Failed to open stable json output: " + json_copy);
                } else {
                  jf2.write(json_payload.data(),
                            static_cast<std::streamsize>(json_payload.size()));
                  jf2.close();
                  if (!jf2) {
                    final_status = absl::InternalError(
                        "Failed to write stable json output: " + json_copy);
                  }
                }
              }

              // Copy TXT
              if (final_status.ok()) {
                std::ofstream out2(txt_copy);
                if (!out2) {
                  final_status = absl::InternalError(
                      "Failed to open stable txt output: " + txt_copy);
                } else {
                  out2 << "User: " << users[i] << "\n";
                  out2 << "Success: " << (rpc->success ? "true" : "false")
                       << "\n";
                  out2 << "Message: " << rpc->message << "\n";
                  out2 << "GLB_Bytes: " << glb_payload.size() << "\n";
                  out2 << "JSON_Bytes: " << json_payload.size() << "\n";
                  out2.close();
                  if (!out2) {
                    final_status = absl::InternalError(
                        "Failed to write stable txt output: " + txt_copy);
                  }
                }
              }

              if (final_status.ok()) {
                GTEST_LOG_(INFO) << "[" << users[i]
                                 << "] Also wrote stable copies:\n"
                                 << "  txt:  " << txt_copy << "\n"
                                 << "  glb:  " << glb_copy << "\n"
                                 << "  json: " << json_copy;
              }
            }

            if (!final_status.ok()) {
              GTEST_LOG_(INFO) << "[" << users[i]
                               << "] Stable output copy failed: "
                               << final_status.ToString();
            }
          }
        } else {
          GTEST_LOG_(INFO) << "[" << users[i] << "] Output write failed: "
                           << final_status.ToString();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mu);
        outcomes[i].status = final_status;
        done++;
        cv.notify_all();
      }
    });
  }

  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::minutes(20);
  int waited_s = 0;
  while (true) {
    std::unique_lock<std::mutex> lock(mu);
    if (done == static_cast<int>(users.size())) break;
    if (cv.wait_for(lock, std::chrono::seconds(1)) == std::cv_status::timeout) {
      waited_s++;
      if (waited_s % 10 == 0) {
        GTEST_LOG_(INFO) << "Waiting for sam3d responses... (" << waited_s
                         << "s)";
      }
    }
    if (std::chrono::steady_clock::now() > deadline) {
      FAIL() << "Timed out waiting for sam3d responses.";
    }
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  GTEST_LOG_(INFO) << "RPC attempts: " << rpc_attempts.load();
  EXPECT_EQ(rpc_attempts.load(), static_cast<int>(users.size()))
      << "Some client threads did not reach the RPC send step.";

  for (size_t i = 0; i < users.size(); ++i) {
    EXPECT_TRUE(outcomes[i].status.ok())
        << users[i] << " failed: " << outcomes[i].status.ToString();
    const std::string content = ReadAll(outcomes[i].output_path);
    EXPECT_FALSE(content.empty()) << "Missing output for " << users[i]
                                  << " at " << outcomes[i].output_path;

    const std::string base_path = JoinPath(out_dir, "result_" + users[i]);
    const std::string glb_path = base_path + "_objmodel.glb";
    const std::string json_path = base_path + "_objinfo.json";

    EXPECT_TRUE(std::filesystem::exists(glb_path))
        << "Missing GLB for " << users[i] << " at " << glb_path;
    EXPECT_TRUE(std::filesystem::exists(json_path))
        << "Missing JSON for " << users[i] << " at " << json_path;
  }
}

}  // namespace spatialblend::sam3d
