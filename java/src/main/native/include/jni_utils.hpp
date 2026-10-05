/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cudf::jni {
constexpr jint MINIMUM_JNI_VERSION = JNI_VERSION_1_6;

/**
 * @brief Helper to convert a pointer to a jlong.
 *
 * This is useful when, for instance, converting a cudf::column pointer
 * to a jlong, for use in JNI.
 */
template <typename T>
jlong ptr_as_jlong(T* ptr)
{ return reinterpret_cast<jlong>(ptr); }

/**
 * @brief Helper to release the data held by a unique_ptr, and return
 * the pointer as a jlong.
 */
template <typename T>
jlong release_as_jlong(std::unique_ptr<T>&& ptr)
{ return ptr_as_jlong(ptr.release()); }

/**
 * @brief Helper to release the data held by a unique_ptr, and return
 * the pointer as a jlong.
 */
template <typename T>
jlong release_as_jlong(std::unique_ptr<T>& ptr)
{ return release_as_jlong(std::move(ptr)); }

class native_jdoubleArray_accessor {
 public:
  jdouble* getArrayElements(JNIEnv* const env, jdoubleArray arr) const
  { return env->GetDoubleArrayElements(arr, NULL); }

  jdoubleArray newArray(JNIEnv* const env, int len) const { return env->NewDoubleArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jdoubleArray jarr, int start, int len, jdouble const* arr) const
  { env->SetDoubleArrayRegion(jarr, start, len, arr); }

  void releaseArrayElements(JNIEnv* const env, jdoubleArray jarr, jdouble* arr, jint mode) const
  { env->ReleaseDoubleArrayElements(jarr, arr, mode); }
};

class native_jlongArray_accessor {
 public:
  jlong* getArrayElements(JNIEnv* const env, jlongArray arr) const
  { return env->GetLongArrayElements(arr, NULL); }

  jlongArray newArray(JNIEnv* const env, int len) const { return env->NewLongArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jlongArray jarr, int start, int len, jlong const* arr) const
  { env->SetLongArrayRegion(jarr, start, len, arr); }

  void releaseArrayElements(JNIEnv* const env, jlongArray jarr, jlong* arr, jint mode) const
  { env->ReleaseLongArrayElements(jarr, arr, mode); }
};

class native_jintArray_accessor {
 public:
  jint* getArrayElements(JNIEnv* const env, jintArray arr) const
  { return env->GetIntArrayElements(arr, NULL); }

  jintArray newArray(JNIEnv* const env, int len) const { return env->NewIntArray(len); }

  void setArrayRegion(JNIEnv* const env, jintArray jarr, int start, int len, jint const* arr) const
  { env->SetIntArrayRegion(jarr, start, len, arr); }

  void releaseArrayElements(JNIEnv* const env, jintArray jarr, jint* arr, jint mode) const
  { env->ReleaseIntArrayElements(jarr, arr, mode); }
};

class native_jbyteArray_accessor {
 public:
  jbyte* getArrayElements(JNIEnv* const env, jbyteArray arr) const
  { return env->GetByteArrayElements(arr, NULL); }

  jbyteArray newArray(JNIEnv* const env, int len) const { return env->NewByteArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jbyteArray jarr, int start, int len, jbyte const* arr) const
  { env->SetByteArrayRegion(jarr, start, len, arr); }

  void releaseArrayElements(JNIEnv* const env, jbyteArray jarr, jbyte* arr, jint mode) const
  { env->ReleaseByteArrayElements(jarr, arr, mode); }
};

class native_jbooleanArray_accessor {
 public:
  jboolean* getArrayElements(JNIEnv* const env, jbooleanArray arr) const
  { return env->GetBooleanArrayElements(arr, NULL); }

  jbooleanArray newArray(JNIEnv* const env, int len) const { return env->NewBooleanArray(len); }

  void setArrayRegion(
    JNIEnv* const env, jbooleanArray jarr, int start, int len, jboolean const* arr) const
  { env->SetBooleanArrayRegion(jarr, start, len, arr); }

  void releaseArrayElements(JNIEnv* const env, jbooleanArray jarr, jboolean* arr, jint mode) const
  { env->ReleaseBooleanArrayElements(jarr, arr, mode); }
};

/**
 * @brief RAII for java arrays to be sure it is handled correctly.
 *
 * By default any changes to the array will be committed back when
 * the destructor is called unless cancel is called first.
 */
template <typename N_TYPE, typename J_ARRAY_TYPE, typename ACCESSOR>
class native_jArray {
 private:
  ACCESSOR access{};
  JNIEnv* const env;
  J_ARRAY_TYPE orig;
  int len;
  mutable N_TYPE* data_ptr;

  void init_data_ptr() const
  {
    if (orig != nullptr && data_ptr == nullptr) {
      data_ptr = access.getArrayElements(env, orig);
      check_java_exception(env);
    }
  }

 public:
  native_jArray(native_jArray const&)            = delete;
  native_jArray& operator=(native_jArray const&) = delete;

  native_jArray(JNIEnv* const env, J_ARRAY_TYPE orig) : env(env), orig(orig), len(0), data_ptr(NULL)
  {
    if (orig != NULL) {
      len = env->GetArrayLength(orig);
      check_java_exception(env);
    }
  }

  native_jArray(JNIEnv* const env, int len)
    : env(env), orig(access.newArray(env, len)), len(len), data_ptr(NULL)
  { check_java_exception(env); }

  native_jArray(JNIEnv* const env, N_TYPE const* arr, int len)
    : env(env), orig(access.newArray(env, len)), len(len), data_ptr(NULL)
  {
    check_java_exception(env);
    access.setArrayRegion(env, orig, 0, len, arr);
    check_java_exception(env);
  }

  native_jArray(JNIEnv* const env, std::vector<N_TYPE> const& arr)
    : env(env), orig(access.newArray(env, arr.size())), len(arr.size()), data_ptr(NULL)
  {
    check_java_exception(env);
    access.setArrayRegion(env, orig, 0, len, arr.data());
    check_java_exception(env);
  }

  bool is_null() const noexcept { return orig == NULL; }

  int size() const noexcept { return len; }

  N_TYPE operator[](int index) const
  {
    if (orig == NULL) { throw_java_exception(env, NPE_EXCEPTION_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= len) {
      throw_java_exception(env, INDEX_OOB_EXCEPTION_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  N_TYPE& operator[](int index)
  {
    if (orig == NULL) { throw_java_exception(env, NPE_EXCEPTION_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= len) {
      throw_java_exception(env, INDEX_OOB_EXCEPTION_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  N_TYPE const* data() const
  {
    init_data_ptr();
    return data_ptr;
  }

  N_TYPE* data()
  {
    init_data_ptr();
    return data_ptr;
  }

  N_TYPE const* begin() const { return data(); }

  N_TYPE* begin() { return data(); }

  N_TYPE const* end() const { return data() + size(); }

  N_TYPE* end() { return data() + size(); }

  J_ARRAY_TYPE get_jArray() const { return orig; }

  /**
   * @brief Conversion to std::vector
   *
   * @tparam target_t Target data type
   * @return std::vector<target_t> Vector with the copied contents
   */
  template <typename target_t = N_TYPE>
  std::vector<target_t> to_vector() const
  {
    std::vector<target_t> ret;
    ret.reserve(size());
    std::copy(begin(), end(), std::back_inserter(ret));
    return ret;
  }

  /**
   * @brief if data has been written back into this array, don't commit
   * it.
   */
  void cancel()
  {
    if (data_ptr != NULL && orig != NULL) {
      access.releaseArrayElements(env, orig, data_ptr, JNI_ABORT);
      data_ptr = NULL;
    }
  }

  void commit()
  {
    if (data_ptr != NULL && orig != NULL) {
      access.releaseArrayElements(env, orig, data_ptr, 0);
      data_ptr = NULL;
    }
  }

  ~native_jArray() { commit(); }
};

using native_jdoubleArray = native_jArray<jdouble, jdoubleArray, native_jdoubleArray_accessor>;
using native_jlongArray   = native_jArray<jlong, jlongArray, native_jlongArray_accessor>;
using native_jintArray    = native_jArray<jint, jintArray, native_jintArray_accessor>;
using native_jbyteArray   = native_jArray<jbyte, jbyteArray, native_jbyteArray_accessor>;

/**
 * @brief Specialization of native_jArray for jboolean
 *
 * This class adds special support for conversion to std::vector<X>, where the element
 * value is chosen depending on the jboolean value.
 */
struct native_jbooleanArray
  : native_jArray<jboolean, jbooleanArray, native_jbooleanArray_accessor> {
  native_jbooleanArray(JNIEnv* const env, jbooleanArray orig)
    : native_jArray<jboolean, jbooleanArray, native_jbooleanArray_accessor>(env, orig)
  {
  }

  native_jbooleanArray(native_jbooleanArray const&)            = delete;
  native_jbooleanArray& operator=(native_jbooleanArray const&) = delete;

  template <typename target_t>
  std::vector<target_t> transform_if_else(target_t const& if_true, target_t const& if_false) const
  {
    std::vector<target_t> ret;
    ret.reserve(size());
    std::transform(begin(), end(), std::back_inserter(ret), [&](jboolean const& b) {
      return b ? if_true : if_false;
    });
    return ret;
  }
};

/**
 * @brief wrapper around native_jlongArray to make it take pointers instead.
 *
 * By default any changes to the array will be committed back when
 * the destructor is called unless cancel is called first.
 */
template <typename T>
class native_jpointerArray {
 private:
  native_jlongArray wrapped;
  JNIEnv* const env;

 public:
  native_jpointerArray(native_jpointerArray const&)            = delete;
  native_jpointerArray& operator=(native_jpointerArray const&) = delete;

  native_jpointerArray(JNIEnv* const env, jlongArray orig) : wrapped(env, orig), env(env) {}

  native_jpointerArray(JNIEnv* const env, int len) : wrapped(env, len), env(env) {}

  native_jpointerArray(JNIEnv* const env, T* arr, int len) : wrapped(env, arr, len), env(env) {}

  bool is_null() const noexcept { return wrapped.is_null(); }

  int size() const noexcept { return wrapped.size(); }

  T* operator[](int index) const
  {
    if (data() == NULL) { throw_java_exception(env, NPE_EXCEPTION_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= wrapped.size()) {
      throw_java_exception(env, INDEX_OOB_EXCEPTION_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  T*& operator[](int index)
  {
    if (data() == NULL) { throw_java_exception(env, NPE_EXCEPTION_CLASS, "pointer is NULL"); }
    if (index < 0 || index >= wrapped.size()) {
      throw_java_exception(env, INDEX_OOB_EXCEPTION_CLASS, "NOT IN BOUNDS");
    }
    return data()[index];
  }

  T* const* data() const { return reinterpret_cast<T* const*>(wrapped.data()); }

  T** data() { return reinterpret_cast<T**>(wrapped.data()); }

  T* const* begin() const { return data(); }
  T* const* end() const { return data() + size(); }

  jlongArray get_jArray() const { return wrapped.get_jArray(); }

  void assert_no_nulls() const
  {
    if (std::any_of(data(), data() + size(), [](T* const ptr) { return ptr == nullptr; })) {
      throw_java_exception(env, NPE_EXCEPTION_CLASS, "pointer is NULL");
    }
  }

  /**
   * @brief Convert from `T*[]` to `vector<T>`.
   */
  std::vector<T> get_dereferenced() const
  {
    assert_no_nulls();
    auto ret = std::vector<T>{};
    ret.reserve(size());
    std::transform(
      data(), data() + size(), std::back_inserter(ret), [](T* const& p) { return *p; });
    return ret;
  }

  /**
   * @brief if data has been written back into this array, don't commit
   * it.
   */
  void cancel() { wrapped.cancel(); }

  void commit() { wrapped.commit(); }
};

/**
 * @brief wrapper around native_jlongArray to hold pointers that are deleted
 * if not released, like std::unique_ptr.
 *
 * By default any changes to the array will be committed back when
 * released unless cancel is called first.
 */
template <typename T, typename D = std::default_delete<T>>
class unique_jpointerArray {
 private:
  std::unique_ptr<native_jpointerArray<T>> wrapped;
  D del;

 public:
  unique_jpointerArray(unique_jpointerArray const&)            = delete;
  unique_jpointerArray& operator=(unique_jpointerArray const&) = delete;

  unique_jpointerArray(JNIEnv* const env, jlongArray orig)
    : wrapped(new native_jpointerArray<T>(env, orig))
  {
  }

  unique_jpointerArray(JNIEnv* const env, jlongArray orig, D const& del)
    : wrapped(new native_jpointerArray<T>(env, orig)), del(del)
  {
  }

  unique_jpointerArray(JNIEnv* const env, int len) : wrapped(new native_jpointerArray<T>(env, len))
  {
  }

  unique_jpointerArray(JNIEnv* const env, int len, D const& del)
    : wrapped(new native_jpointerArray<T>(env, len)), del(del)
  {
  }

  unique_jpointerArray(JNIEnv* const env, T* arr, int len)
    : wrapped(new native_jpointerArray<T>(env, arr, len))
  {
  }

  unique_jpointerArray(JNIEnv* const env, T* arr, int len, D const& del)
    : wrapped(new native_jpointerArray<T>(env, arr, len)), del(del)
  {
  }

  bool is_null() const noexcept { return wrapped == NULL || wrapped->is_null(); }

  int size() const noexcept { return wrapped == NULL ? 0 : wrapped->size(); }

  void reset(int index, T* new_ptr = NULL)
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    T* old = (*wrapped)[index];
    if (old != new_ptr) {
      (*wrapped)[index] = new_ptr;
      del(old);
    }
  }

  T* get(int index)
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    return (*wrapped)[index];
  }

  T* const* get()
  {
    if (wrapped == NULL) { throw std::logic_error("using unique_jpointerArray after release"); }
    return wrapped->data();
  }

  jlongArray release()
  {
    if (wrapped == NULL) { return NULL; }
    wrapped->commit();
    jlongArray ret = wrapped->get_jArray();
    wrapped.reset(NULL);
    return ret;
  }

  ~unique_jpointerArray()
  {
    if (wrapped != NULL) {
      for (int i = 0; i < wrapped->size(); i++) {
        reset(i, NULL);
      }
    }
  }
};

/**
 * @brief RAII for jstring to be sure it is handled correctly.
 */
class native_jstring {
 private:
  JNIEnv* env;
  jstring orig;
  mutable char const* cstr;
  mutable size_t cstr_length;
  mutable bool converted = false;

  // Rewrite the "modified UTF-8" (CESU-8) bytes returned by GetStringUTFChars into
  // standard UTF-8. JNI encodes the NUL code point as the two bytes 0xC0 0x80, and code
  // points above U+FFFF as a 6-byte UTF-16 surrogate pair rather than a 4-byte sequence;
  // the rest of libcudf expects standard UTF-8, so re-encode just those two cases. Every
  // other byte is already valid UTF-8 and is copied verbatim, and both rewrites only ever
  // shrink the string, so the result never exceeds the input length. The replacement
  // buffer is allocated lazily on the first byte that needs rewriting. Returns true iff a
  // rewrite occurred, in which case cstr owns a new[] buffer and cstr_length is updated;
  // otherwise cstr is left pointing at the buffer returned by JNI.
  bool convert_cesu8() const
  {
    char* utf8_cstr    = nullptr;
    char* utf8_current = nullptr;
    auto* s            = reinterpret_cast<unsigned char const*>(cstr);
    for (size_t i = 0; i < cstr_length; ++i) {
      bool is_nul            = (i + 1 < cstr_length && s[i] == 0xC0 && s[i + 1] == 0x80);
      bool is_surrogate_pair = (i + 5 < cstr_length && s[i] == 0xED && (s[i + 1] & 0xF0) == 0xA0 &&
                                s[i + 3] == 0xED && (s[i + 4] & 0xF0) == 0xB0);
      if (utf8_cstr == nullptr && (is_nul || is_surrogate_pair)) {
        // First byte needing a rewrite: allocate the output buffer (never larger than
        // the input) and copy the already-scanned, unmodified prefix.
        utf8_cstr = new char[cstr_length + 1];
        std::memcpy(utf8_cstr, cstr, i);
        utf8_current = utf8_cstr + i;
      }
      if (utf8_current == nullptr) continue;  // no rewrite seen yet; keep scanning
      if (is_nul) {
        *utf8_current++ = '\x00';
        ++i;  // consume the second byte of the 0xC0 0x80 pair
      } else if (is_surrogate_pair) {
        unsigned int hi        = (s[i + 1] & 0x0F) << 6 | (s[i + 2] & 0x3F);
        unsigned int lo        = (s[i + 4] & 0x0F) << 6 | (s[i + 5] & 0x3F);
        unsigned int codepoint = 0x10000 + (hi << 10 | lo);
        *utf8_current++        = static_cast<char>(0xF0 | (codepoint >> 18 & 0x07));
        *utf8_current++        = static_cast<char>(0x80 | (codepoint >> 12 & 0x3F));
        *utf8_current++        = static_cast<char>(0x80 | (codepoint >> 6 & 0x3F));
        *utf8_current++        = static_cast<char>(0x80 | (codepoint & 0x3F));
        i += 5;  // consume the remaining 5 bytes of the surrogate pair
      } else {
        *utf8_current++ = cstr[i];
      }
    }
    if (utf8_current == nullptr) return false;  // nothing was rewritten; keep JNI's buffer
    cstr_length   = utf8_current - utf8_cstr;
    *utf8_current = '\0';
    env->ReleaseStringUTFChars(orig, cstr);
    cstr = utf8_cstr;
    return true;
  }

  void init_cstr() const
  {
    if (orig != nullptr && cstr == nullptr) {
      cstr_length = env->GetStringUTFLength(orig);
      cstr        = env->GetStringUTFChars(orig, 0);
      check_java_exception(env);
      converted = convert_cesu8();
    }
  }

 public:
  native_jstring(native_jstring const&)            = delete;
  native_jstring& operator=(native_jstring const&) = delete;

  native_jstring(native_jstring&& other) noexcept
    : env(other.env),
      orig(other.orig),
      cstr(other.cstr),
      cstr_length(other.cstr_length),
      converted(other.converted)
  { other.cstr = nullptr; }

  native_jstring(JNIEnv* const env, jstring orig)
    : env(env), orig(orig), cstr(nullptr), cstr_length(0), converted(false)
  {
  }

  native_jstring& operator=(native_jstring const&& other)
  {
    if (orig != nullptr && cstr != nullptr) {
      if (converted) {
        delete[] cstr;
      } else {
        env->ReleaseStringUTFChars(orig, cstr);
      }
    }
    this->env         = other.env;
    this->orig        = other.orig;
    this->cstr        = other.cstr;
    this->cstr_length = other.cstr_length;
    this->converted   = other.converted;
    other.cstr        = nullptr;
    return *this;
  }

  bool is_null() const noexcept { return orig == nullptr; }

  char const* get() const
  {
    init_cstr();
    return cstr;
  }

  size_t size_bytes() const
  {
    init_cstr();
    return cstr_length;
  }

  // Implicit conversions so a native_jstring can be passed directly wherever libcudf wants a
  // std::string_view / std::string. Both use size_bytes() (not strlen), so embedded NULs survive.
  // The string_view aliases this object's buffer and must not outlive the native_jstring.
  operator std::string_view() const
  {
    init_cstr();
    return std::string_view(cstr, cstr_length);
  }

  operator std::string() const
  {
    init_cstr();
    return std::string(cstr, cstr_length);
  }

  bool is_empty() const
  {
    if (cstr != nullptr) {
      return cstr_length <= 0;
    } else if (orig != nullptr) {
      jsize len = env->GetStringLength(orig);
      check_java_exception(env);
      return len <= 0;
    }
    return true;
  }

  jstring get_jstring() const { return orig; }

  ~native_jstring()
  {
    if (orig != nullptr && cstr != nullptr) {
      if (converted) {
        delete[] cstr;
      } else {
        env->ReleaseStringUTFChars(orig, cstr);
      }
    }
  }
};

/**
 * @brief jobjectArray wrapper to make accessing it more convenient.
 */
template <typename T>
class native_jobjectArray {
 private:
  JNIEnv* const env;
  jobjectArray orig;
  int len;

 public:
  native_jobjectArray(JNIEnv* const env, jobjectArray orig) : env(env), orig(orig), len(0)
  {
    if (orig != NULL) {
      len = env->GetArrayLength(orig);
      check_java_exception(env);
    }
  }

  bool is_null() const noexcept { return orig == NULL; }

  int size() const noexcept { return len; }

  T operator[](int index) const { return get(index); }

  T get(int index) const
  {
    if (orig == NULL) {
      throw_java_exception(env, NPE_EXCEPTION_CLASS, "jobjectArray pointer is NULL");
    }
    T ret = static_cast<T>(env->GetObjectArrayElement(orig, index));
    check_java_exception(env);
    return ret;
  }

  void set(int index, T const& val)
  {
    if (orig == NULL) {
      throw_java_exception(env, NPE_EXCEPTION_CLASS, "jobjectArray pointer is NULL");
    }
    env->SetObjectArrayElement(orig, index, val);
    check_java_exception(env);
  }

  jobjectArray wrapped() { return orig; }
};

/**
 * @brief jobjectArray wrapper to make accessing strings safe through RAII
 * and convenient.
 */
class native_jstringArray {
 private:
  JNIEnv* const env;
  native_jobjectArray<jstring> arr;
  mutable std::vector<native_jstring> cache;
  mutable std::vector<std::string> cpp_cache;
  mutable std::vector<char const*> c_cache;

  void init_cache() const
  {
    if (!arr.is_null() && cache.empty()) {
      int size = this->size();
      cache.reserve(size);
      for (int i = 0; i < size; i++) {
        cache.push_back(native_jstring(env, arr.get(i)));
      }
    }
  }

  void init_c_cache() const
  {
    if (!arr.is_null() && c_cache.empty()) {
      init_cache();
      int size = this->size();
      c_cache.reserve(size);
      for (int i = 0; i < size; i++) {
        c_cache.push_back(cache[i].get());
      }
    }
  }

  void init_cpp_cache() const
  {
    if (!arr.is_null() && cpp_cache.empty()) {
      init_cache();
      int size = this->size();
      cpp_cache.reserve(size);
      for (int i = 0; i < size; i++) {
        cpp_cache.push_back(cache[i]);
      }
    }
  }

  void update_caches(int index, jstring val)
  {
    if (!cache.empty()) {
      cache[index] = native_jstring(env, val);
      if (!c_cache.empty()) { c_cache[index] = cache[index].get(); }

      if (!cpp_cache.empty()) { cpp_cache[index] = cache[index]; }
    } else if (!c_cache.empty() || !cpp_cache.empty()) {
      // Illegal state
      throw std::logic_error("CACHING IS MESSED UP");
    }
  }

 public:
  native_jstringArray(JNIEnv* const env, jobjectArray orig) : env(env), arr(env, orig) {}

  bool is_null() const noexcept { return arr.is_null(); }

  int size() const noexcept { return arr.size(); }

  native_jstring& operator[](int index) const { return get(index); }

  native_jstring& get(int index) const
  {
    if (arr.is_null()) {
      throw_java_exception(env, cudf::jni::NPE_EXCEPTION_CLASS, "jstringArray pointer is NULL");
    }
    init_cache();
    return cache[index];
  }

  char const** as_c_array() const
  {
    init_c_cache();
    return c_cache.data();
  }

  std::vector<std::string> as_cpp_vector() const
  {
    init_cpp_cache();
    return cpp_cache;
  }

  void set(int index, jstring val)
  {
    arr.set(index, val);
    update_caches(index, val);
  }

  void set(int index, native_jstring const& val)
  {
    arr.set(index, val.get_jstring());
    update_caches(index, val.get_jstring());
  }

  void set(int index, char const* val)
  {
    jstring str = env->NewStringUTF(val);
    check_java_exception(env);
    arr.set(index, str);
    update_caches(index, str);
  }
};

inline auto add_global_ref(JNIEnv* env, jobject jobj)
{
  auto new_global_ref = env->NewGlobalRef(jobj);
  if (new_global_ref == nullptr) { throw cudf::jni::jni_exception("global ref"); }
  return new_global_ref;
}

inline nullptr_t del_global_ref(JNIEnv* env, jobject jobj)
{
  if (jobj != nullptr) { env->DeleteGlobalRef(jobj); }
  return nullptr;
}
}  // namespace cudf::jni
