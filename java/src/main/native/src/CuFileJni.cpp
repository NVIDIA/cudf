/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

#include <cudf/utilities/error.hpp>

#include <cufile.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>

namespace {

/**
 * @brief Get the error description based on the CUDA driver error code.
 *
 * @param cu_result CUDA driver error code.
 * @return Description for the error.
 */
char const* GetCuErrorString(CUresult cu_result)
{
  char const* description;
  if (cuGetErrorName(cu_result, &description) != CUDA_SUCCESS) description = "unknown cuda error";
  return description;
}

/**
 * @brief Get the error description based on the integer error code.
 *
 * cuFile APIs return both cuFile specific error codes as well as POSIX error codes for ease of use.
 *
 * @param error_code Integer error code.
 * @return Description of the error.
 */
std::string cuFileGetErrorString(int error_code)
{
  return IS_CUFILE_ERR(error_code) ? std::string(CUFILE_ERRSTR(error_code))
                                   : std::string(std::strerror(error_code));
}

/**
 * @brief Get the error description based on the cuFile return status.
 *
 * @param status cuFile return status.
 * @return Description of the error.
 */
std::string cuFileGetErrorString(CUfileError_t status)
{
  std::string error = cuFileGetErrorString(status.err);
  if (IS_CUDA_ERR(status)) { error.append(".").append(GetCuErrorString(status.cu_err)); }
  return error;
}

/**
 * @brief RAII wrapper for the cuFile driver.
 */
class cufile_driver {
 public:
  /** @brief Construct a new driver instance by opening the cuFile driver. */
  cufile_driver()
  {
    auto const status = cuFileDriverOpen();
    CUDF_EXPECTS(status.err == CU_FILE_SUCCESS,
                 "Failed to initialize cuFile driver: " + cuFileGetErrorString(status));
  }

  // Disable copy (and move) semantics.
  cufile_driver(cufile_driver const&)            = delete;
  cufile_driver& operator=(cufile_driver const&) = delete;

  /** @brief Destroy the driver instance by closing the cuFile driver. */
  ~cufile_driver() { cuFileDriverClose(); }
};

/** @brief RAII wrapper for a device buffer used by cuFile. */
class cufile_buffer {
 public:
  /**
   * @brief Construct a new cuFile buffer.
   *
   * @param device_pointer Pointer to the device buffer.
   * @param size The size of the allocated device buffer.
   * @param register_buffer Whether to register the buffer with cuFile. This should only be set to
   * true if this buffer is being reused and is 4KiB aligned.
   */
  cufile_buffer(void* device_pointer, std::size_t size, bool register_buffer = false)
    : device_pointer_{device_pointer}, size_{size}, register_buffer_{register_buffer}
  {
    if (register_buffer_) {
      auto const status = cuFileBufRegister(device_pointer_, size_, 0);
      CUDF_EXPECTS(status.err == CU_FILE_SUCCESS,
                   "Failed to register cuFile buffer: " + cuFileGetErrorString(status));
    }
  }

  // Disable copy (and move) semantics.
  cufile_buffer(cufile_buffer const&)            = delete;
  cufile_buffer& operator=(cufile_buffer const&) = delete;

  /** @brief Destroy the buffer by de-registering it if necessary. */
  ~cufile_buffer()
  {
    if (register_buffer_) { cuFileBufDeregister(device_pointer_); }
  }

  /**
   * @brief Get the pointer to the underlying device buffer.
   *
   * @return Pointer to the device buffer.
   */
  void* device_pointer() const { return device_pointer_; }

  /**
   * @brief Get the size of the underlying device buffer.
   *
   * @return The size of the device buffer.
   */
  std::size_t size() const { return size_; }

 private:
  /// Pointer to the device buffer.
  void* device_pointer_;
  /// Size of the device buffer.
  std::size_t size_;
  /// Whether to register the buffer with cuFile.
  bool register_buffer_;
};

/** @brief RAII wrapper for a file descriptor and the corresponding cuFile handle. */
class cufile_file {
 public:
  /**
   * @brief Construct a file wrapper.
   *
   * Should not be called directly; use the following factory methods instead.
   *
   * @param file_descriptor A valid file descriptor.
   */
  explicit cufile_file(int file_descriptor) : file_descriptor_{file_descriptor}
  {
    CUfileDescr_t cufile_descriptor{CU_FILE_HANDLE_TYPE_OPAQUE_FD, file_descriptor_};
    auto const status = cuFileHandleRegister(&cufile_handle_, &cufile_descriptor);
    if (status.err != CU_FILE_SUCCESS) {
      close(file_descriptor_);
      CUDF_FAIL("Failed to register cuFile handle: " + cuFileGetErrorString(status));
    }
  }

  /**
   * @brief Factory method to create a file wrapper for reading.
   *
   * @param path Absolute path of the file to read from.
   * @return std::unique_ptr<cufile_file> for reading.
   */
  static auto make_reader(char const* path)
  {
    auto const file_descriptor = open(path, O_RDONLY | O_DIRECT);
    CUDF_EXPECTS(file_descriptor >= 0,
                 "Failed to open file to read: " + cuFileGetErrorString(errno));
    return std::make_unique<cufile_file>(file_descriptor);
  }

  /**
   * @brief Factory method to create a file wrapper for writing.
   *
   * @param path Absolute path of the file to write to.
   * @return std::unique_ptr<cufile_file> for writing.
   */
  static auto make_writer(char const* path)
  {
    auto const file_descriptor = open(path, O_CREAT | O_WRONLY | O_DIRECT, S_IRUSR | S_IWUSR);
    CUDF_EXPECTS(file_descriptor >= 0,
                 "Failed to open file to write: " + cuFileGetErrorString(errno));
    return std::make_unique<cufile_file>(file_descriptor);
  }

  // Disable copy (and move) semantics.
  cufile_file(cufile_file const&)            = delete;
  cufile_file& operator=(cufile_file const&) = delete;

  /** @brief Destroy the file wrapper by de-registering the cuFile handle and closing the file. */
  ~cufile_file()
  {
    cuFileHandleDeregister(cufile_handle_);
    close(file_descriptor_);
  }

  /**
   * @brief Read the file into a device buffer.
   *
   * @param buffer Device buffer to read the file content into.
   * @param file_offset Starting offset from which to read the file.
   */
  void read(cufile_buffer const& buffer, std::size_t file_offset) const
  {
    auto const status =
      cuFileRead(cufile_handle_, buffer.device_pointer(), buffer.size(), file_offset, 0);

    CUDF_EXPECTS(status >= 0,
                 "Failed to read file into buffer: " +
                   cuFileGetErrorString(IS_CUFILE_ERR(status) ? status : errno));

    CUDF_EXPECTS(static_cast<std::size_t>(status) == buffer.size(),
                 "Size of bytes read is different from buffer size");
  }

  /**
   * @brief Write a device buffer to the file.
   *
   * @param buffer The device buffer to write.
   * @param size The number of bytes to write.
   * @param file_offset Starting offset from which to write the buffer.
   */
  void write(cufile_buffer const& buffer, std::size_t size, std::size_t file_offset)
  {
    auto const status = cuFileWrite(cufile_handle_, buffer.device_pointer(), size, file_offset, 0);

    CUDF_EXPECTS(status >= 0,
                 "Failed to write buffer to file: " +
                   cuFileGetErrorString(IS_CUFILE_ERR(status) ? status : errno));

    CUDF_EXPECTS(static_cast<std::size_t>(status) == size,
                 "Size of bytes written is different from the specified size");
  }

  /**
   * @brief Append a device buffer to the file.
   *
   * @param buffer The device buffer to append.
   * @param size The number of bytes to append.
   * @return The file offset from which the buffer was appended.
   */
  std::size_t append(cufile_buffer const& buffer, std::size_t size)
  {
    struct stat stat_buffer;
    auto const status = fstat(file_descriptor_, &stat_buffer);
    CUDF_EXPECTS(status >= 0,
                 "Failed to get file status for appending: " + cuFileGetErrorString(errno));

    auto const file_offset = static_cast<std::size_t>(stat_buffer.st_size);
    write(buffer, size, file_offset);
    return file_offset;
  }

 private:
  /// The underlying file descriptor.
  int file_descriptor_;
  /// The registered cuFile handle.
  CUfileHandle_t cufile_handle_{};
};

}  // anonymous namespace

extern "C" {

/**
 * @brief Create a new cuFile driver wrapper.
 *
 * @param env The JNI environment.
 * @return Pointer address to the new driver wrapper instance.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFileDriver_create(JNIEnv* env, jclass)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return reinterpret_cast<jlong>(new cufile_driver());
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Destroy the given cuFile driver wrapper.
 *
 * @param env The JNI environment.
 * @param pointer Pointer address to the driver wrapper instance.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFileDriver_destroy(JNIEnv* env, jclass, jlong pointer)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<cufile_driver>(pointer);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Create a new cuFile buffer wrapper.
 *
 * @param env The JNI environment.
 * @param device_pointer Pointer address to the device buffer.
 * @param size The size of the device buffer.
 * @param register_buffer If true, register the cuFile buffer.
 * @return Pointer address to the new buffer wrapper instance.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFileBuffer_create(
  JNIEnv* env, jclass, jlong device_pointer, jlong size, jboolean register_buffer)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* buffer =
      new cufile_buffer(reinterpret_cast<void*>(device_pointer), size, register_buffer);
    return reinterpret_cast<jlong>(buffer);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Destroy the given cuFile buffer wrapper.
 *
 * @param env The JNI environment.
 * @param pointer Pointer address to the buffer wrapper instance.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFileBuffer_destroy(JNIEnv* env, jclass, jlong pointer)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<cufile_buffer>(pointer);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Create a new cuFile file handle wrapper for reading.
 *
 * @param env The JNI environment.
 * @param path The file path to read from.
 * @return Pointer address to the new file handle wrapper instance.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFileReadHandle_create(JNIEnv* env,
                                                                    jclass,
                                                                    jstring path)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto file = cufile_file::make_reader(env->GetStringUTFChars(path, nullptr));
    return reinterpret_cast<jlong>(file.release());
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Read the content into the specified buffer.
 *
 * @param env The JNI environment.
 * @param file Pointer to the cuFile file object.
 * @param file_offset The file offset from which to read.
 * @param buffer Pointer to the cuFile buffer object.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFileReadHandle_readIntoBuffer(
  JNIEnv* env, jclass, jlong file, jlong file_offset, jlong buffer)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* file_ptr   = reinterpret_cast<cufile_file*>(file);
    auto* buffer_ptr = reinterpret_cast<cufile_buffer*>(buffer);
    file_ptr->read(*buffer_ptr, file_offset);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Create a new cuFile file handle wrapper for writing.
 *
 * @param env The JNI environment.
 * @param path The file path to write to.
 * @return Pointer address to the new file handle wrapper instance.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFileWriteHandle_create(JNIEnv* env,
                                                                     jclass,
                                                                     jstring path)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto file = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
    return reinterpret_cast<jlong>(file.release());
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Write the content of the specified buffer into the file.
 *
 * @param env The JNI environment.
 * @param file Pointer to the cuFile file object.
 * @param file_offset The file offset from which to write.
 * @param buffer Pointer to the cuFile buffer object.
 * @param size Number of bytes to write.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFileWriteHandle_writeFromBuffer(
  JNIEnv* env, jclass, jlong file, jlong file_offset, jlong buffer, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* file_ptr   = reinterpret_cast<cufile_file*>(file);
    auto* buffer_ptr = reinterpret_cast<cufile_buffer*>(buffer);
    file_ptr->write(*buffer_ptr, size, file_offset);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Append the content of the specified buffer into the file.
 *
 * @param env The JNI environment.
 * @param file Pointer to the cuFile file object.
 * @param buffer Pointer to the cuFile buffer object.
 * @param size Number of bytes to append
 * @return The file offset from which the buffer was appended.
 */
JNIEXPORT long JNICALL Java_ai_rapids_cudf_CuFileWriteHandle_appendFromBuffer(
  JNIEnv* env, jclass, jlong file, jlong buffer, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* file_ptr   = reinterpret_cast<cufile_file*>(file);
    auto* buffer_ptr = reinterpret_cast<cufile_buffer*>(buffer);
    return file_ptr->append(*buffer_ptr, size);
  }
  JNI_CATCH(env, -1);
}

/**
 * @brief Destroy the given cuFile file handle wrapper.
 *
 * @param env The JNI environment.
 * @param pointer Pointer address to the file handle wrapper instance.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFileHandle_destroy(JNIEnv* env, jclass, jlong pointer)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::safe_delete<cufile_file>(pointer);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Write a device buffer into a given file path.
 *
 * @param env The JNI environment.
 * @param path Absolute path of the file to copy the buffer to.
 * @param file_offset The file offset from which the buffer was written.
 * @param device_pointer Pointer address to the device buffer.
 * @param size Number of bytes to write.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_writeToFile(
  JNIEnv* env, jclass, jstring path, jlong file_offset, jlong device_pointer, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cufile_buffer buffer{reinterpret_cast<void*>(device_pointer), static_cast<std::size_t>(size)};
    auto writer = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
    writer->write(buffer, size, file_offset);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Append a device buffer into a given file path.
 *
 * @param env The JNI environment.
 * @param path Absolute path of the file to copy the buffer to.
 * @param device_pointer Pointer address to the device buffer.
 * @param size Number of bytes to append.
 * @return The file offset from which the buffer was appended.
 */
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_CuFile_appendToFile(
  JNIEnv* env, jclass, jstring path, jlong device_pointer, jlong size)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cufile_buffer buffer{reinterpret_cast<void*>(device_pointer), static_cast<std::size_t>(size)};
    auto writer = cufile_file::make_writer(env->GetStringUTFChars(path, nullptr));
    return writer->append(buffer, size);
  }
  JNI_CATCH(env, -1);
}

/**
 * @brief Read from a given file path into a device buffer.
 *
 * @param env The JNI environment.
 * @param device_pointer Pointer address to the device buffer.
 * @param size The size of the device buffer.
 * @param path Absolute path of the file to copy from.
 * @param file_offset The file offset from which to copy content.
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_CuFile_readFromFile(
  JNIEnv* env, jclass, jlong device_pointer, jlong size, jstring path, jlong file_offset)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cufile_buffer buffer{reinterpret_cast<void*>(device_pointer), static_cast<std::size_t>(size)};
    auto const reader = cufile_file::make_reader(env->GetStringUTFChars(path, nullptr));
    reader->read(buffer, file_offset);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
