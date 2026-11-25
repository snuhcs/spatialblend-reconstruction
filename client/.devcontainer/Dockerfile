FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install basic dependencies.
# spatialblend-session is a client-only library — no GUI packages needed.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gdb \
    curl \
    git \
    gnupg \
    unzip \
    wget \
    zip \
    fish

# Install bazel
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Get buildifier to format bazel-related files.
RUN wget https://github.com/bazelbuild/buildtools/releases/download/v6.4.0/buildifier-linux-amd64 -O /usr/local/bin/buildifier && \
    chmod +x /usr/local/bin/buildifier

# Install clang-format
RUN apt-get install -y clang-format

# Install OpenJDK with retry logic and alternative mirrors
RUN apt-get install -y openjdk-17-jdk

# Find the actual Java installation directory and create symlink for consistent JAVA_HOME
RUN JAVA_HOME_DIR=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "Actual Java directory: $JAVA_HOME_DIR" && \
    if [ "$JAVA_HOME_DIR" != "/usr/lib/jvm/java-17-openjdk-amd64" ]; then \
    ln -sfn "$JAVA_HOME_DIR" /usr/lib/jvm/java-17-openjdk-amd64; \
    fi
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

ENV ANDROID_DEV_HOME=/android
RUN mkdir -p ${ANDROID_DEV_HOME}

# Install Android SDK.
ENV ANDROID_CMD_TOOLS_FILENAME=commandlinetools-linux-6200805_latest.zip
ENV ANDROID_SDK_URL=https://dl.google.com/android/repository/
ENV ANDROID_API_LEVEL=28
ENV ANDROID_NDK_API_LEVEL=21

# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION=30.0.0
# NDK 26.1 LTS - required for rules_android_ndk 0.1.3 (needs NDK 25b+)
ENV ANDROID_NDK_VERSION=26.1.10909125
ENV ANDROID_SDK_HOME=${ANDROID_DEV_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL}${ANDROID_CMD_TOOLS_FILENAME} && \
    unzip ${ANDROID_CMD_TOOLS_FILENAME} && \
    rm ${ANDROID_CMD_TOOLS_FILENAME}
RUN yes | ${ANDROID_DEV_HOME}/tools/bin/sdkmanager --licenses --sdk_root=${ANDROID_DEV_HOME}
RUN ${ANDROID_DEV_HOME}/tools/bin/sdkmanager --sdk_root=${ANDROID_DEV_HOME} "platforms;android-${ANDROID_API_LEVEL}" "build-tools;${ANDROID_BUILD_TOOLS_VERSION}"
RUN ${ANDROID_DEV_HOME}/tools/bin/sdkmanager --sdk_root=${ANDROID_DEV_HOME} "ndk;${ANDROID_NDK_VERSION}"
# Find the actual installed NDK version and set ANDROID_NDK_HOME
RUN mkdir -p /etc/profile.d && \
    if [ -d "${ANDROID_DEV_HOME}/ndk" ]; then \
    INSTALLED_NDK=$(ls -d ${ANDROID_DEV_HOME}/ndk/* 2>/dev/null | head -1) && \
    if [ -n "$INSTALLED_NDK" ]; then \
    echo "Found installed NDK: $INSTALLED_NDK" && \
    NDK_VERSION=$(basename "$INSTALLED_NDK") && \
    echo "Using NDK version: $NDK_VERSION" && \
    echo "ANDROID_NDK_HOME=$INSTALLED_NDK" >> /etc/environment && \
    echo "export ANDROID_NDK_HOME=$INSTALLED_NDK" >> /etc/profile.d/android.sh && \
    chmod +x /etc/profile.d/android.sh; \
    else \
    echo "No NDK found in ${ANDROID_DEV_HOME}/ndk"; \
    fi; \
    else \
    echo "NDK directory does not exist"; \
    fi

# Set default ANDROID_NDK_HOME - will be overridden by actual installation if found
ENV ANDROID_NDK_HOME=${ANDROID_DEV_HOME}/ndk/${ANDROID_NDK_VERSION}
ENV PATH=${PATH}:$ANDROID_SDK_HOME/tools:$ANDROID_SDK_HOME/platform-tools:$ANDROID_NDK_HOME

# Make android ndk executable to all users.
RUN chmod -R go=u ${ANDROID_DEV_HOME}

# Set ANDROID_HOME to the android dev home.
ENV ANDROID_HOME=${ANDROID_DEV_HOME}


# Devcontainer metadata for Cursor/VSCode attach
LABEL devcontainer.metadata='[{"remoteUser":"root","workspaceFolder":"/workspace/spatialblend-session"}]'

WORKDIR /workspace/spatialblend-session

# Verify installations
RUN bazel version && java -version
