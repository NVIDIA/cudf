# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# See
# https://github.com/cython/cython/blob/master/Cython/Utility/CppSupport.cpp
# for the original Cython exception handler.
cdef extern from *:
    """
    #include <Python.h>
    #include <cucascade/io/io_errors.hpp>
    #include <ios>
    #include <stdexcept>

    namespace {

    /**
     * @brief Exception handler to map C++ exceptions to Python ones in Cython
     *
     * This exception handler extends the base exception handler provided by
     * Cython. In addition to the exceptions that Cython itself supports, it maps
     * the exceptions that cuCascade throws to specific Python exceptions.
     *
     * Since this function interoperates with Python's exception state, it
     * does not throw any C++ exceptions.
     */
    void libcucascade_exception_handler()
    {
      try {
        if (PyErr_Occurred())
          ;  // let latest Python exn pass through and ignore the current one
        throw;
      } catch (const std::bad_alloc& exn) {
        PyErr_SetString(PyExc_MemoryError, exn.what());
      } catch (const std::bad_cast& exn) {
        PyErr_SetString(PyExc_TypeError, exn.what());
      } catch (const std::domain_error& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::invalid_argument& exn) {
        PyErr_SetString(PyExc_ValueError, exn.what());
      } catch (const std::ios_base::failure& exn) {
        PyErr_SetString(PyExc_IOError, exn.what());
      } catch (const std::out_of_range& exn) {
        PyErr_SetString(PyExc_IndexError, exn.what());
      } catch (const std::overflow_error& exn) {
        PyErr_SetString(PyExc_OverflowError, exn.what());
      } catch (const std::range_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const std::underflow_error& exn) {
        PyErr_SetString(PyExc_ArithmeticError, exn.what());
      } catch (const cucascade::io::credential_error& exn) {
        // Catch subclasses before their parent (runtime_error)
        PyErr_SetString(PyExc_PermissionError, exn.what());
      } catch (const cucascade::io::object_changed_error& exn) {
        PyErr_SetString(PyExc_OSError, exn.what());
      } catch (const std::exception& exn) {
        // The default catch-all case.
        PyErr_SetString(PyExc_RuntimeError, exn.what());
      } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
      }
    }

    }  // anonymous namespace
    """
    cdef void libcucascade_exception_handler()
