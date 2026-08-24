---
name: build-test-cudf-java
description: Build and test cudf Java bindings (cudf-java) inside a cudf devcontainer. Use when the user asks to build, compile, or test Java code in the cudf repository.
---

# Build and Test cudf-java

## Prerequisites

### JDK and Maven

Ensure JDK (17+) and Maven are installed:

```bash
java -version && mvn --version
```

If either is missing, detect the OS and install using the appropriate package manager. Hints:

- **Debian/Ubuntu** (`apt`): `sudo apt-get update -qq && sudo apt-get install -y -qq default-jdk maven`
- **Fedora/RHEL** (`dnf`): `sudo dnf install -y java-17-openjdk-devel maven`
- **Arch** (`pacman`): `sudo pacman -S --noconfirm jdk17-openjdk maven`
- **macOS** (`brew`): `brew install openjdk@17 maven`

Detect the OS by checking which package manager is available (e.g. `command -v apt-get`, `command -v dnf`, etc.) and use the matching command.

Run all Maven commands below from the `java/` directory.

## Choose a libcudf provider

Prefer the installed nightly package when the changes are limited to Java or JNI and the user has not requested a libcudf source rebuild:

- [Build JNI against an installed nightly libcudf](references/nightly-libcudf.md), then continue at [Running Java tests](#running-java-tests).

Use a local libcudf source build when the changes affect libcudf, require unreleased libcudf headers or symbols, no compatible nightly exists, or the user explicitly requests a source build:

- [Build JNI against a local libcudf build](references/local-libcudf.md), then continue at [Running Java tests](#running-java-tests).

## Running Java tests

**Always run `mvn install $MVN_COMMON_OPTS -DskipTests` first** before running tests. The `mvn test` goal re-triggers the cmake/native build step. If `target/cmake-build` already contains fully built artifacts from a prior `mvn install`, this is an incremental no-op. But if the cmake-build directory was cleaned or is missing, `mvn test` may hit a race condition where the linker tries to link `libcudfjni.so` before `libarrow.a` is fully built, causing a `cannot find libarrow.a` error. If this happens, re-run `mvn install $MVN_COMMON_OPTS -DskipTests` to rebuild the native code cleanly, then retry `mvn test`.

### All tests

```bash
mvn test $MVN_COMMON_OPTS
```

The POM routes tests that need special assertion or fork behavior to dedicated Surefire executions.

### Discovering tests

Java test sources are at `src/test/java/ai/rapids/cudf/`. Use `find` or `glob` to discover test classes:

```bash
find src/test/java -name "*Test.java" | head -20
```

### Specific test class

```bash
mvn surefire:test@main-tests $MVN_COMMON_OPTS \
  -Dtest="<ClassName>"
```

### Specific test method

```bash
mvn surefire:test@main-tests $MVN_COMMON_OPTS \
  -Dtest="<ClassName>#<testName>"
```

Use the named `main-tests` execution for specific ordinary tests so `-Dtest` does not affect the separately configured executions.

### Tests with special configuration

Invoke tests that require special configuration through their dedicated executions:

```bash
mvn surefire:test@non-empty-null-test $MVN_COMMON_OPTS
mvn surefire:test@fatal-cuda-test $MVN_COMMON_OPTS
mvn surefire:test@native-deps-loader-test $MVN_COMMON_OPTS
```
