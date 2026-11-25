// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#ifndef SESSION_NETWORK_UTIL_ANDROID_NETWORK_H_
#define SESSION_NETWORK_UTIL_ANDROID_NETWORK_H_

#ifdef __ANDROID__

#include <jni.h>

#include <string>

namespace spatialblend {
namespace network {
namespace util {

// Android-specific network utilities using JNI to call ConnectivityManager
class AndroidNetworkUtils {
 public:
  // Initialize JNI environment (call this from JNI_OnLoad)
  static void Initialize(JNIEnv* env);

  // Initialize JNI with Android application context
  static void InitializeWithContext(JNIEnv* env, jobject context);

  // Initialize JNI and auto-detect Android context (no Java-side initialization
  // needed)
  static void InitializeWithAutoContext(JNIEnv* env);

  // Get the device's IP address using Android ConnectivityManager
  // Returns empty string if not available or if not on Android
  static std::string GetDeviceIPAddress();

  // Check if network is available
  static bool IsNetworkAvailable();

  // Get active network type (WiFi, Mobile, etc.)
  static std::string GetActiveNetworkType();

  // Test access methods (for testing purposes)
  static JNIEnv* GetJNIEnv();
  static bool IsInitialized();

 private:
  // Helper methods for JNI implementation
  static jobject GetApplicationContext(JNIEnv* env);
  static jobject GetApplicationContextFromActivityThread(JNIEnv* env);
  static jobject GetWifiManager(JNIEnv* env, jobject context);
  static jobject GetDhcpInfo(JNIEnv* env, jobject wifiManager);
  static std::string ConvertIntToIpString(jint ipAddress);

  static JavaVM* jvm_;
  static jobject application_context_;
  static jclass connectivity_manager_class_;
  static jclass network_info_class_;
  static jclass context_class_;
  static jclass wifi_manager_class_;
  static jclass dhcp_info_class_;
  static jmethodID get_system_service_method_;
  static jmethodID get_active_network_info_method_;
  static jmethodID is_connected_method_;
  static jmethodID get_type_name_method_;
  static jmethodID get_extra_info_method_;
  static jmethodID get_dhcp_info_method_;
  static jfieldID ip_address_field_;
};

// JNI_OnLoad is defined in android_network.cc and is automatically called
// when the native library is loaded. It initializes the JNI environment
// (JavaVM reference, class references, and method IDs).
// The application context is auto-detected or can be set explicitly via
// InitializeWithContext() if needed for full functionality.

}  // namespace util
}  // namespace network
}  // namespace spatialblend

#endif  // __ANDROID__

#endif  // SESSION_NETWORK_UTIL_ANDROID_NETWORK_H_
