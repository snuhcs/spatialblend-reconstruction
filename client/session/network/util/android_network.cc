// Copyright (c) 2026 Seoul National University
// SPDX-License-Identifier: MIT

#include "session/network/util/android_network.h"

#ifdef __ANDROID__

#include <android/log.h>
#include <arpa/inet.h>
#include <jni.h>
#include <netinet/in.h>

#include "session/util/logger.h"

namespace spatialblend {
namespace network {
namespace util {

// Static member definitions
JavaVM* AndroidNetworkUtils::jvm_ = nullptr;
jobject AndroidNetworkUtils::application_context_ = nullptr;
jclass AndroidNetworkUtils::connectivity_manager_class_ = nullptr;
jclass AndroidNetworkUtils::network_info_class_ = nullptr;
jclass AndroidNetworkUtils::context_class_ = nullptr;
jclass AndroidNetworkUtils::wifi_manager_class_ = nullptr;
jclass AndroidNetworkUtils::dhcp_info_class_ = nullptr;
jmethodID AndroidNetworkUtils::get_system_service_method_ = nullptr;
jmethodID AndroidNetworkUtils::get_active_network_info_method_ = nullptr;
jmethodID AndroidNetworkUtils::is_connected_method_ = nullptr;
jmethodID AndroidNetworkUtils::get_type_name_method_ = nullptr;
jmethodID AndroidNetworkUtils::get_extra_info_method_ = nullptr;
jmethodID AndroidNetworkUtils::get_dhcp_info_method_ = nullptr;
jfieldID AndroidNetworkUtils::ip_address_field_ = nullptr;

#define CHECK_JNI_EXCEPTION(env, log_message) \
  do { \
    if (env->ExceptionCheck()) { \
      SPATIALBLEND_LOG(LogSeverity::kError, "Exception in JNI:  ", log_message); \
      env->ExceptionDescribe(); \
      env->ExceptionClear(); \
      return; \
    } \
  } while (0)

#define CHECK_JNI_EXCEPTION_NULL(env, log_message) \
  do { \
    if (env->ExceptionCheck()) { \
      SPATIALBLEND_LOG(LogSeverity::kError, "Exception null in JNI:  ", log_message); \
      env->ExceptionDescribe(); \
      env->ExceptionClear(); \
      return nullptr; \
    } \
  } while (0)

void AndroidNetworkUtils::Initialize(JNIEnv* env) {
  // Check if JNI classes and methods are already initialized
  // Don't check application_context_ here as it's set separately
  if (jvm_ && connectivity_manager_class_ && network_info_class_ &&
      context_class_ && wifi_manager_class_ && dhcp_info_class_ &&
      get_system_service_method_ && get_active_network_info_method_ &&
      is_connected_method_ && get_type_name_method_ &&
      get_extra_info_method_ && get_dhcp_info_method_ &&
      ip_address_field_) {
    return;
  }

  // Store JVM reference
  if (!jvm_) {
    env->GetJavaVM(&jvm_);
  }

  // Get Context class
  jclass temp_class;
  if (!context_class_) {
    temp_class = env->FindClass("android/content/Context");
    CHECK_JNI_EXCEPTION(env, "Failed to find Context class");
    context_class_ = (jclass)env->NewGlobalRef(temp_class);
    CHECK_JNI_EXCEPTION(env, "Failed to create global reference to Context class");
    env->DeleteLocalRef(temp_class);
  }

  // Get ConnectivityManager class
  if (!connectivity_manager_class_) {
    temp_class = env->FindClass("android/net/ConnectivityManager");
    CHECK_JNI_EXCEPTION(env, "Failed to find ConnectivityManager class");
    connectivity_manager_class_ = (jclass)env->NewGlobalRef(temp_class);
    CHECK_JNI_EXCEPTION(env, "Failed to create NewGlobalRef for ConnectivityManager");
    env->DeleteLocalRef(temp_class);
  }

  // Get NetworkInfo class
  if (!network_info_class_) {
    temp_class = env->FindClass("android/net/NetworkInfo");
    CHECK_JNI_EXCEPTION(env, "Failed to find NetworkInfo class");
    network_info_class_ = (jclass)env->NewGlobalRef(temp_class);
    CHECK_JNI_EXCEPTION(env, "Failed to create NewGlobalRef for NetworkInfo");
    env->DeleteLocalRef(temp_class);
  }

  // Get WifiManager class
  if (!wifi_manager_class_) {
    temp_class = env->FindClass("android/net/wifi/WifiManager");
    CHECK_JNI_EXCEPTION(env, "Failed to find WifiManager class");
    wifi_manager_class_ = (jclass)env->NewGlobalRef(temp_class);
    CHECK_JNI_EXCEPTION(env, "Failed to create NewGlobalRef for WifiManager");
    env->DeleteLocalRef(temp_class);
  }

  // Get DhcpInfo class
  if (!dhcp_info_class_) {
    temp_class = env->FindClass("android/net/DhcpInfo");
    CHECK_JNI_EXCEPTION(env, "Failed to find DhcpInfo class");
    dhcp_info_class_ = (jclass)env->NewGlobalRef(temp_class);
    CHECK_JNI_EXCEPTION(env, "Failed to create NewGlobalRef for DhcpInfo");
    env->DeleteLocalRef(temp_class);
  }

  // Get method IDs
  if (!get_system_service_method_) {
    get_system_service_method_ =
        env->GetMethodID(context_class_, "getSystemService",
                         "(Ljava/lang/String;)Ljava/lang/Object;");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for getSystemService");
  }
  if (!get_active_network_info_method_) {
    get_active_network_info_method_ =
        env->GetMethodID(connectivity_manager_class_, "getActiveNetworkInfo",
                         "()Landroid/net/NetworkInfo;");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for getActiveNetworkInfo");
  }
  if (!is_connected_method_) {
    is_connected_method_ =
        env->GetMethodID(network_info_class_, "isConnected", "()Z");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for isConnected");
  }
  if (!get_type_name_method_) {
    get_type_name_method_ = env->GetMethodID(network_info_class_, "getTypeName",
                                             "()Ljava/lang/String;");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for getTypeName");
  }
  if (!get_extra_info_method_) {
    get_extra_info_method_ = env->GetMethodID(network_info_class_, "getExtraInfo",
                                              "()Ljava/lang/String;");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for getExtraInfo");
  }
  if (!get_dhcp_info_method_) {
    get_dhcp_info_method_ = env->GetMethodID(wifi_manager_class_, "getDhcpInfo",
                                             "()Landroid/net/DhcpInfo;");
    CHECK_JNI_EXCEPTION(env, "Failed to get method ID for getDhcpInfo");
  }
  if (!ip_address_field_) {
    ip_address_field_ = env->GetFieldID(dhcp_info_class_, "ipAddress", "I");
    CHECK_JNI_EXCEPTION(env, "Failed to get field ID for ipAddress");
  }
}

void AndroidNetworkUtils::InitializeWithContext(JNIEnv* env, jobject context) {
  if (!context) {
    SPATIALBLEND_LOG(LogSeverity::kError,
                  "Context is null in InitializeWithContext");
    return;
  }

  // Store application context globally
  application_context_ = env->NewGlobalRef(context);
  CHECK_JNI_EXCEPTION(env, "Failed to create global reference to application context");

  // Initialize JNI
  Initialize(env);
}

void AndroidNetworkUtils::InitializeWithAutoContext(JNIEnv* env) {
  // Initialize JNI first
  Initialize(env);

  // Try to get application context directly from ActivityThread
  jobject context = GetApplicationContextFromActivityThread(env);
  if (context) {
    application_context_ = env->NewGlobalRef(context);
    CHECK_JNI_EXCEPTION(env, "Failed to create global reference to application context");
  } else {
    SPATIALBLEND_LOG(LogSeverity::kError,
                  "Failed to get application context from ActivityThread");
  }
}

JNIEnv* AndroidNetworkUtils::GetJNIEnv() {
  if (!jvm_) {
    return nullptr;
  }

  JNIEnv* env;
  int result = jvm_->GetEnv((void**)&env, JNI_VERSION_1_6);

  if (result == JNI_EDETACHED) {
    // Thread is not attached, attach it
    result = jvm_->AttachCurrentThread(&env, nullptr);
    if (result != JNI_OK) {
      SPATIALBLEND_LOG(LogSeverity::kError,
                    "Failed to attach current thread to JVM");
      return nullptr;
    }
    return env;
  } else if (result != JNI_OK) {
    SPATIALBLEND_LOG(LogSeverity::kError, "Failed to get JNI environment");
    return nullptr;
  }

  return env;
}

bool AndroidNetworkUtils::IsInitialized() {
  return jvm_ && application_context_ && connectivity_manager_class_ &&
         network_info_class_ && context_class_ && wifi_manager_class_ &&
         dhcp_info_class_ && get_system_service_method_ &&
         get_active_network_info_method_ && is_connected_method_ &&
         get_type_name_method_ && get_extra_info_method_ &&
         get_dhcp_info_method_ && ip_address_field_;
}

std::string AndroidNetworkUtils::GetDeviceIPAddress() {
#ifdef __ANDROID__
  JNIEnv* env = GetJNIEnv();
  if (!env) {
    return "127.0.0.1";
  }

  // Try auto-initialization if not already initialized
  if (!IsInitialized()) {
    InitializeWithAutoContext(env);
  }

  if (!IsInitialized()) {
    return "127.0.0.1";
  }

  try {
    // Get the current application context
    // This should now be auto-detected from ActivityThread
    jobject context = GetApplicationContext(env);
    if (!context) {
      SPATIALBLEND_LOG(
          LogSeverity::kWarning,
          "Application context not available, falling back to localhost");
      return "127.0.0.1";
    }

    // Get WifiManager
    jobject wifiManager = GetWifiManager(env, context);
    if (!wifiManager) {
      SPATIALBLEND_LOG(LogSeverity::kWarning,
                    "Could not get WifiManager, falling back to localhost");
      return "127.0.0.1";
    }

    // Get DHCP info
    jobject dhcpInfo = GetDhcpInfo(env, wifiManager);
    if (!dhcpInfo) {
      SPATIALBLEND_LOG(LogSeverity::kWarning,
                    "Could not get DHCP info, falling back to localhost");
      return "127.0.0.1";
    }

    // Get IP address as integer
    jint ipAddress = env->GetIntField(dhcpInfo, ip_address_field_);
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
      SPATIALBLEND_LOG(LogSeverity::kWarning,
                    "Exception getting IP address, falling back to localhost");
      return "127.0.0.1";
    }

    // Convert integer IP to string
    std::string ipString = ConvertIntToIpString(ipAddress);
    if (ipString.empty() || ipString == "0.0.0.0") {
      SPATIALBLEND_LOG(LogSeverity::kWarning,
                    "Invalid IP address (%d), falling back to localhost",
                    ipAddress);
      return "127.0.0.1";
    }

    return ipString;

  } catch (...) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "Exception in GetDeviceIPAddress, falling back to localhost");
    return "127.0.0.1";
  }
#else
  return "";  // Not on Android
#endif
}

bool AndroidNetworkUtils::IsNetworkAvailable() {
#ifdef __ANDROID__
  JNIEnv* env = GetJNIEnv();
  if (!env || !IsInitialized()) {
    return true;  // Assume available if we can't check
  }

  try {
    // Similar to GetDeviceIPAddress, we'd need proper context access
    // For now, assume network is available
    return true;
  } catch (...) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "Exception in IsNetworkAvailable, assuming available");
    return true;
  }
#else
  return true;  // Not on Android, assume available
#endif
}

std::string AndroidNetworkUtils::GetActiveNetworkType() {
#ifdef __ANDROID__
  JNIEnv* env = GetJNIEnv();
  if (!env || !IsInitialized()) {
    return "unknown";
  }

  try {
    // Similar implementation would go here
    return "unknown";
  } catch (...) {
    SPATIALBLEND_LOG(LogSeverity::kWarning, "Exception in GetActiveNetworkType");
    return "unknown";
  }
#else
  return "unknown";  // Not on Android
#endif
}

// Helper methods for JNI implementation
jobject AndroidNetworkUtils::GetApplicationContext(JNIEnv* env) {
  // Return the stored application context
  return application_context_;
}

jobject AndroidNetworkUtils::GetApplicationContextFromActivityThread(
    JNIEnv* env) {
  // Get ActivityThread class
  jclass activityThreadClass = env->FindClass("android/app/ActivityThread");
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to find ActivityThread class");
  if (!activityThreadClass) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "Could not find ActivityThread class");
    return nullptr;
  }

  // Get currentActivityThread() method
  jmethodID currentActivityThreadMethod =
      env->GetStaticMethodID(activityThreadClass, "currentActivityThread",
                              "()Landroid/app/ActivityThread;");
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to get currentActivityThread method");
  if (!currentActivityThreadMethod) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "Could not find currentActivityThread method");
    return nullptr;
  }

  // Call currentActivityThread()
  jobject activityThread = env->CallStaticObjectMethod(
      activityThreadClass, currentActivityThreadMethod);
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to call currentActivityThread");
  if (!activityThread) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "currentActivityThread() returned null");
    return nullptr;
  }

  // Get getApplication() method
  jmethodID getApplicationMethod = env->GetMethodID(
      activityThreadClass, "getApplication", "()Landroid/app/Application;");
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to get getApplication method");
  if (!getApplicationMethod) {
    SPATIALBLEND_LOG(LogSeverity::kWarning,
                  "Could not find getApplication method");
    return nullptr;
  }

  // Call getApplication()
  jobject application =
      env->CallObjectMethod(activityThread, getApplicationMethod);
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to call getApplication");
  if (!application) {
    SPATIALBLEND_LOG(LogSeverity::kWarning, "getApplication() returned null");
    return nullptr;
  }

  env->DeleteLocalRef(activityThread);
  env->DeleteLocalRef(activityThreadClass);
  return application;
}

jobject AndroidNetworkUtils::GetWifiManager(JNIEnv* env, jobject context) {
  if (!context) return nullptr;

  // Get WIFI_SERVICE constant
  jfieldID wifiServiceField = env->GetStaticFieldID(
      context_class_, "WIFI_SERVICE", "Ljava/lang/String;");
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to get WIFI_SERVICE field");
  if (!wifiServiceField) return nullptr;

  jstring wifiService =
      (jstring)env->GetStaticObjectField(context_class_, wifiServiceField);
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to get WIFI_SERVICE");
  if (!wifiService) return nullptr;

  // Call getSystemService
  jobject wifiManager =
      env->CallObjectMethod(context, get_system_service_method_, wifiService);
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to call getSystemService for WIFI_SERVICE");

  env->DeleteLocalRef(wifiService);
  return wifiManager;
}

jobject AndroidNetworkUtils::GetDhcpInfo(JNIEnv* env, jobject wifiManager) {
  if (!wifiManager) return nullptr;

  jobject dhcpInfo = env->CallObjectMethod(wifiManager, get_dhcp_info_method_);
  CHECK_JNI_EXCEPTION_NULL(env, "Failed to call getDhcpInfo");

  return dhcpInfo;
}

std::string AndroidNetworkUtils::ConvertIntToIpString(jint ipAddress) {
  if (ipAddress == 0) return "";

  // Convert integer IP to string format
  unsigned char bytes[4];
  bytes[0] = (ipAddress >> 0) & 0xFF;
  bytes[1] = (ipAddress >> 8) & 0xFF;
  bytes[2] = (ipAddress >> 16) & 0xFF;
  bytes[3] = (ipAddress >> 24) & 0xFF;

  char ipString[16];
  snprintf(ipString, sizeof(ipString), "%d.%d.%d.%d", bytes[0], bytes[1],
           bytes[2], bytes[3]);

  return std::string(ipString);
}

}  // namespace util
}  // namespace network
}  // namespace spatialblend

// JNI_OnLoad is called when the native library is loaded
// This is the standard entry point for JNI initialization
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
  SPATIALBLEND_LOG(spatialblend::LogSeverity::kInfo, "JNI_OnLoad Called");
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Initialize JNI references and method IDs
  // This will also store the JavaVM reference
  spatialblend::network::util::AndroidNetworkUtils::Initialize(env);

  if (env->ExceptionCheck()) {
    SPATIALBLEND_LOG(spatialblend::LogSeverity::kError, "Exception in JNI_OnLoad Initialize");
    env->ExceptionDescribe();
    env->ExceptionClear();
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

#endif  // __ANDROID__
