// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
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

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_write.h"

#include "session/inference/distributed/sam3_seg_client.h"
#include "session/inference/distributed/sam3d_obj_client.h"
#include "session/inference/inference_client.h"

namespace spatialblend::distributed {
namespace {

// ---------------------------------------------------------------------------
// Environment / path utilities (adapted from sam3d_remote_test.cc)
// ---------------------------------------------------------------------------

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
  return (dir.back() == '/' || dir.back() == '\\') ? dir + file
                                                    : dir + "/" + file;
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

std::string GetWorkspaceName() {
  std::string ws = GetEnvOrEmpty("TEST_WORKSPACE");
  return ws.empty() ? std::string("_main") : ws;
}

std::string GetRunfilesDir() {
  std::string v = GetEnvOrEmpty("RUNFILES_DIR");
  if (!v.empty()) return v;
  v = GetEnvOrEmpty("TEST_SRCDIR");
  if (!v.empty()) return v;
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
  if (runfiles_dir.empty()) return workspace_relative_path;
  return (std::filesystem::path(runfiles_dir) / ws / workspace_relative_path)
      .string();
}

std::vector<std::string> ListFilesByExt(const std::string& dir,
                                        const std::vector<std::string>& exts) {
  std::vector<std::string> out;
  std::error_code ec;
  if (!std::filesystem::exists(dir, ec)) return out;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const std::string ext = entry.path().extension().string();
    for (const auto& e : exts) {
      if (ext.size() == e.size()) {
        bool match = true;
        for (size_t i = 0; i < ext.size(); ++i) {
          if (std::tolower(ext[i]) != std::tolower(e[i])) {
            match = false;
            break;
          }
        }
        if (match) {
          out.push_back(entry.path().string());
          break;
        }
      }
    }
  }
  return out;
}

std::string ReadFileBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// Reads 2D points from file; returns flat [x1,y1,0, x2,y2,0, ...]
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
    ss >> x >> comma >> y;
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

// ---------------------------------------------------------------------------
// Mask overlay helper
// ---------------------------------------------------------------------------

// Overlays a binary mask (0=bg, 1=fg) onto the original image and saves PNG.
// Matches Python: fg pixels get RGBA [30, 144, 255, 128] blended at alpha=128/255.
// Returns true on success.
bool SaveMaskOverlay(const std::string& image_bytes,
                     const std::string& mask_data,
                     int32_t mask_w, int32_t mask_h,
                     const std::string& out_path) {
  // Decode original image
  int img_w = 0, img_h = 0, img_channels = 0;
  unsigned char* pixels = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(image_bytes.data()),
      static_cast<int>(image_bytes.size()),
      &img_w, &img_h, &img_channels, 3 /*force RGB*/);
  if (!pixels) return false;

  // Build RGBA output buffer
  const int total_pixels = img_w * img_h;
  std::vector<unsigned char> out(total_pixels * 4);
  for (int i = 0; i < total_pixels; ++i) {
    out[i * 4 + 0] = pixels[i * 3 + 0];
    out[i * 4 + 1] = pixels[i * 3 + 1];
    out[i * 4 + 2] = pixels[i * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  stbi_image_free(pixels);

  // Overlay: for each fg pixel in mask, alpha-blend [30, 144, 255] at a=128/255
  // Nearest-neighbor mapping from mask coords to image coords.
  const float alpha = 128.0f / 255.0f;
  for (int my = 0; my < mask_h; ++my) {
    for (int mx = 0; mx < mask_w; ++mx) {
      const int mask_idx = my * mask_w + mx;
      if (mask_idx >= static_cast<int>(mask_data.size())) continue;
      if (static_cast<unsigned char>(mask_data[mask_idx]) != 1) continue;

      // Nearest-neighbor map to image pixel
      const int ix = static_cast<int>(static_cast<float>(mx) * img_w / mask_w);
      const int iy = static_cast<int>(static_cast<float>(my) * img_h / mask_h);
      if (ix < 0 || ix >= img_w || iy < 0 || iy >= img_h) continue;

      const int oi = (iy * img_w + ix) * 4;
      out[oi + 0] = static_cast<unsigned char>(out[oi + 0] * (1.0f - alpha) + 30  * alpha);
      out[oi + 1] = static_cast<unsigned char>(out[oi + 1] * (1.0f - alpha) + 144 * alpha);
      out[oi + 2] = static_cast<unsigned char>(out[oi + 2] * (1.0f - alpha) + 255 * alpha);
    }
  }

  // Write PNG (RGBA)
  const int ok = stbi_write_png(out_path.c_str(), img_w, img_h, 4,
                                out.data(), img_w * 4);
  return ok != 0;
}

// ---------------------------------------------------------------------------
// Per-user result
// ---------------------------------------------------------------------------

struct UserOutcome {
  absl::Status status;
  // Number of successful Segment steps (should equal total point count)
  int segment_steps_ok = 0;
  int segment_steps_total = 0;
  // Final Reconstruct success
  bool reconstruct_ok = false;
  // Output file paths
  std::string glb_path;
  std::string json_path;
};

// ---------------------------------------------------------------------------
// Per-user worker: Segment loop then Reconstruct
// ---------------------------------------------------------------------------

UserOutcome RunUserPipeline(const std::string& user,
                            const std::string& addr,
                            const std::string& image_path,
                            const std::string& points_path,
                            const std::string& out_dir) {
  UserOutcome outcome;

  // ---- Load test data ----
  const std::string image_bytes = ReadFileBinary(image_path);
  if (image_bytes.empty()) {
    outcome.status = absl::InternalError("Failed to read image: " + image_path);
    return outcome;
  }
  const std::vector<float> all_params = ReadPoints2DFileAsFlat(points_path);
  if (all_params.empty()) {
    outcome.status =
        absl::InternalError("Failed to read points: " + points_path);
    return outcome;
  }
  const size_t num_points = all_params.size() / 3;
  outcome.segment_steps_total = static_cast<int>(num_points);

  GTEST_LOG_(INFO) << "[" << user << "] image=" << image_path
                   << " (" << image_bytes.size() << "B)"
                   << " points=" << num_points;

  // ---- Connect Segment client ----
  auto seg_client = std::make_shared<Sam3SegClient>();
  {
    absl::Status st = seg_client->ConnectToAddress(addr);
    if (!st.ok()) {
      outcome.status = st;
      GTEST_LOG_(INFO) << "[" << user << "] Sam3SegClient connect failed: "
                       << st.ToString();
      return outcome;
    }
  }
  GTEST_LOG_(INFO) << "[" << user << "] Sam3SegClient connected";

  // ---- Connect Reconstruct client ----
  auto obj_client = std::make_shared<Sam3dObjClient>();
  {
    absl::Status st = obj_client->ConnectToAddress(addr);
    if (!st.ok()) {
      outcome.status = st;
      GTEST_LOG_(INFO) << "[" << user << "] Sam3dObjClient connect failed: "
                       << st.ToString();
      return outcome;
    }
  }
  GTEST_LOG_(INFO) << "[" << user << "] Sam3dObjClient connected";

  // ---- Phase 1: Segment loop (1 point -> 2 points -> ... -> N points) ----
  std::string last_mask_data;
  int32_t last_mask_width = 0;
  int32_t last_mask_height = 0;

  for (size_t n = 1; n <= num_points; ++n) {
    GTEST_LOG_(INFO) << "[" << user << "] Segment step " << n << "/"
                     << num_points;

    auto fut = seg_client->Process(
        user + "_seg", image_bytes,
        all_params.data(), n * 3  // first n points -> n*3 floats
    );
    absl::StatusOr<inference::InferenceResult> rpc = fut.get();

    if (!rpc.ok()) {
      GTEST_LOG_(INFO) << "[" << user << "] Segment step " << n
                       << " gRPC error: " << rpc.status().ToString();
      outcome.status = rpc.status();
      return outcome;
    }
    if (!rpc->success) {
      GTEST_LOG_(INFO) << "[" << user << "] Segment step " << n
                       << " server error: " << rpc->message;
      outcome.status = absl::InternalError(rpc->message);
      return outcome;
    }

    GTEST_LOG_(INFO) << "[" << user << "] Segment step " << n << " OK | "
                     << "mask=" << rpc->mask_width << "x" << rpc->mask_height
                     << " (" << rpc->output_data.size() << "B) | "
                     << rpc->message;

    outcome.segment_steps_ok++;

    // Save raw mask for debugging
    const std::string mask_path =
        JoinPath(out_dir, user + "_mask_step" +
                               std::to_string(n) + ".raw");
    {
      std::ofstream mf(mask_path, std::ios::binary);
      if (mf) {
        mf.write(rpc->output_data.data(),
                 static_cast<std::streamsize>(rpc->output_data.size()));
      }
    }

    // Save overlay PNG
    const std::string overlay_path =
        JoinPath(out_dir, user + "_overlay_step" + std::to_string(n) + ".png");
    if (SaveMaskOverlay(image_bytes, rpc->output_data,
                        rpc->mask_width, rpc->mask_height, overlay_path)) {
      GTEST_LOG_(INFO) << "[" << user << "] Saved overlay: " << overlay_path;
    } else {
      GTEST_LOG_(INFO) << "[" << user << "] Overlay save failed (step " << n
                       << ")";
    }

    // Keep final mask for Reconstruct
    last_mask_data = rpc->output_data;
    last_mask_width = rpc->mask_width;
    last_mask_height = rpc->mask_height;
  }

  if (last_mask_data.empty()) {
    outcome.status = absl::InternalError("No mask data from Segment phase");
    return outcome;
  }

  // ---- Phase 2: Reconstruct with final mask ----
  GTEST_LOG_(INFO) << "[" << user << "] Reconstruct | mask="
                   << last_mask_width << "x" << last_mask_height;

  auto rec_fut = obj_client->Reconstruct(
      user + "_rec", image_bytes, last_mask_data,
      last_mask_width, last_mask_height);
  absl::StatusOr<inference::InferenceResult> rec = rec_fut.get();

  if (!rec.ok()) {
    GTEST_LOG_(INFO) << "[" << user << "] Reconstruct gRPC error: "
                     << rec.status().ToString();
    outcome.status = rec.status();
    return outcome;
  }
  if (!rec->success) {
    GTEST_LOG_(INFO) << "[" << user << "] Reconstruct server error: "
                     << rec->message;
    outcome.status = absl::InternalError(rec->message);
    return outcome;
  }

  GTEST_LOG_(INFO) << "[" << user << "] Reconstruct OK | glb="
                   << rec->output_data.size() << "B json="
                   << rec->json_data.size() << "B | " << rec->message;

  // Save GLB + JSON
  const std::string ts = TimestampForFilename();
  outcome.glb_path = JoinPath(out_dir, user + "_" + ts + "_objmodel.glb");
  outcome.json_path = JoinPath(out_dir, user + "_" + ts + "_objinfo.json");

  {
    std::ofstream glb(outcome.glb_path, std::ios::binary);
    if (glb) {
      glb.write(rec->output_data.data(),
                static_cast<std::streamsize>(rec->output_data.size()));
    }
  }
  if (!rec->json_data.empty()) {
    std::ofstream jf(outcome.json_path, std::ios::binary);
    if (jf) {
      jf.write(rec->json_data.data(),
               static_cast<std::streamsize>(rec->json_data.size()));
    }
  }

  GTEST_LOG_(INFO) << "[" << user << "] Saved: " << outcome.glb_path;
  GTEST_LOG_(INFO) << "[" << user << "] Saved: " << outcome.json_path;

  outcome.reconstruct_ok = true;
  outcome.status = absl::OkStatus();
  return outcome;
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST(DistributedRemoteTest, MultiUserSegmentThenReconstruct) {
  GTEST_LOG_(INFO) << "=== DistributedRemoteTest starting ===";
  LogEnv("SPATIALBLEND_DISTRIBUTED_ADDR");
  LogEnv("SPATIALBLEND_DISTRIBUTED_TEST_OUTPUT_DIR");
  LogEnv("TEST_TMPDIR");
  LogEnv("TEST_UNDECLARED_OUTPUTS_DIR");
  LogEnv("RUNFILES_DIR");
  LogEnv("TEST_SRCDIR");
  LogEnv("TEST_WORKSPACE");

  const std::string addr = GetEnvOrEmpty("SPATIALBLEND_DISTRIBUTED_ADDR");
  if (addr.empty()) {
    FAIL() << "SPATIALBLEND_DISTRIBUTED_ADDR is not set. "
              "Set it to host:port of a running scan server (see server/).";
  }

  // Output directory: prefer stable dir, then bazel outputs, then tmpdir
  const std::string stable_dir =
      GetEnvOrEmpty("SPATIALBLEND_DISTRIBUTED_TEST_OUTPUT_DIR");
  const std::string outputs_dir = GetEnvOrEmpty("TEST_UNDECLARED_OUTPUTS_DIR");
  const std::string tmpdir = GetEnvOrEmpty("TEST_TMPDIR");
  const std::string out_dir = !stable_dir.empty()   ? stable_dir
                              : !outputs_dir.empty() ? outputs_dir
                              : !tmpdir.empty()      ? tmpdir
                                                     : ".";
  GTEST_LOG_(INFO) << "Output directory: " << out_dir;
  std::filesystem::create_directories(out_dir);

  // Test data from sam3d (shared test data)
  const std::string images_dir =
      ResolveRunfilePath("session/inference/sam3d/test_data/images");
  const std::string points_dir =
      ResolveRunfilePath("session/inference/sam3d/test_data/points");

  GTEST_LOG_(INFO) << "images_dir: " << images_dir;
  GTEST_LOG_(INFO) << "points_dir: " << points_dir;

  const std::vector<std::string> image_paths =
      ListFilesByExt(images_dir, {".png", ".jpg", ".jpeg", ".webp"});
  const std::vector<std::string> point_paths =
      ListFilesByExt(points_dir, {".txt"});

  ASSERT_FALSE(image_paths.empty())
      << "No images found under: " << images_dir;
  ASSERT_FALSE(point_paths.empty())
      << "No point files found under: " << points_dir;

  GTEST_LOG_(INFO) << "Found " << image_paths.size() << " image(s), "
                   << point_paths.size() << " point file(s)";

  // ---- Users ----
  const std::vector<std::string> users = {"Alice", "Bob", "Charlie", "Dave"};

  // Random start delays (sorted ascending) to simulate staggered arrivals
  std::vector<int> delays_s;
  {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(3, 13);
    for (size_t i = 0; i < users.size(); ++i) delays_s.push_back(dist(rng));
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
    GTEST_LOG_(INFO) << "Start delays (s, sorted): " << oss.str();
  }

  // ---- Launch user threads ----
  std::mutex mu;
  std::condition_variable cv;
  int done = 0;
  std::vector<UserOutcome> outcomes(users.size());

  std::vector<std::thread> threads;
  threads.reserve(users.size());

  for (size_t i = 0; i < users.size(); ++i) {
    threads.emplace_back([&, i]() {
      std::this_thread::sleep_for(std::chrono::seconds(delays_s[i]));

      GTEST_LOG_(INFO) << "[" << users[i] << "] thread starting";

      // Pick a random image/points pair (matched by basename)
      const std::string image_path =
          image_paths[PickRandomIndex(image_paths.size())];
      const std::string stem =
          std::filesystem::path(image_path).stem().string();
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
            "No matching points file for image: " + image_path);
        done++;
        cv.notify_all();
        return;
      }

      outcomes[i] =
          RunUserPipeline(users[i], addr, image_path, points_path, out_dir);

      {
        std::lock_guard<std::mutex> lock(mu);
        done++;
        cv.notify_all();
      }
    });
  }

  // ---- Wait for all users (timeout 30 min) ----
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::minutes(30);
  int waited_s = 0;
  while (true) {
    std::unique_lock<std::mutex> lock(mu);
    if (done == static_cast<int>(users.size())) break;
    if (cv.wait_for(lock, std::chrono::seconds(1)) ==
        std::cv_status::timeout) {
      waited_s++;
      if (waited_s % 15 == 0) {
        GTEST_LOG_(INFO) << "Waiting... (" << waited_s << "s, "
                         << done << "/" << users.size() << " done)";
      }
    }
    if (std::chrono::steady_clock::now() > deadline) {
      FAIL() << "Timed out waiting for all users to complete.";
    }
  }

  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  // ---- Assertions ----
  GTEST_LOG_(INFO) << "\n=== Results ===";
  for (size_t i = 0; i < users.size(); ++i) {
    const auto& o = outcomes[i];
    GTEST_LOG_(INFO) << "[" << users[i] << "] "
                     << "status=" << o.status.ToString()
                     << " segment_ok=" << o.segment_steps_ok
                     << "/" << o.segment_steps_total
                     << " reconstruct=" << (o.reconstruct_ok ? "OK" : "FAIL");

    EXPECT_TRUE(o.status.ok())
        << users[i] << " failed: " << o.status.ToString();

    EXPECT_EQ(o.segment_steps_ok, o.segment_steps_total)
        << users[i] << ": not all Segment steps succeeded";

    EXPECT_TRUE(o.reconstruct_ok)
        << users[i] << ": Reconstruct failed";

    if (o.reconstruct_ok) {
      EXPECT_TRUE(std::filesystem::exists(o.glb_path))
          << users[i] << ": GLB file missing: " << o.glb_path;
      EXPECT_GT(std::filesystem::file_size(o.glb_path), 0u)
          << users[i] << ": GLB file is empty";

      EXPECT_TRUE(std::filesystem::exists(o.json_path))
          << users[i] << ": JSON file missing: " << o.json_path;
      EXPECT_GT(std::filesystem::file_size(o.json_path), 0u)
          << users[i] << ": JSON file is empty";
    }
  }
}

}  // namespace
}  // namespace spatialblend::distributed
