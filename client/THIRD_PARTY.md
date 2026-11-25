# Third-party components

## Vendored in this repository

| Path | Component | Licence |
|---|---|---|
| `third_party/stb/` | stb_image v2.30, stb_image_write v1.16 (Sean Barrett) | MIT or public domain (dual), text at the end of each header; used by the manual remote test only, not linked into the library |
| `third_party/rules_android_ndk/windows_fix.patch` | patch to rules_android_ndk | Apache-2.0 (as the patched project) |

## Fetched by Bazel (`MODULE.bazel`), not vendored

| Module | Licence | In the shipped library? |
|---|---|---|
| grpc 1.72.0-pre1 (with upb, utf8_range, address_sorting, envoy_api, xds, cel-spec, protoc-gen-validate, googleapis protos) | Apache-2.0 (bundled parts BSD-3-Clause / MPL-2.0) | yes |
| protobuf 30.0 | BSD-3-Clause | yes |
| abseil-cpp 20250127.0 | Apache-2.0 | yes |
| boringssl | OpenSSL/SSLeay + ISC | yes (through gRPC; also the TLS test) |
| zlib, c-ares, re2, xxhash (through gRPC); utf8_range, upb (in protobuf) | Zlib, MIT, BSD-3-Clause, BSD-2-Clause, MIT, BSD-3-Clause | yes |
| googletest 1.15.2 | BSD-3-Clause | no (tests) |
| rules_pkg, rules_python, rules_android_ndk, platforms | Apache-2.0 | no (build rules) |

`THIRD_PARTY_NOTICES-native.txt` carries the full licence texts of what the libraries link statically and goes
with every binary release.
