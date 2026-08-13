/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a public type whose API is still considered experimental and may change
 * without notice. Code annotated with {@code @Experimental} is not subject to
 * cuDF Java compatibility guarantees.
 *
 * <p>Public nested types of an annotated type inherit the experimental status by
 * association and do not need a separate marker.
 */
@Documented
@Retention(RetentionPolicy.CLASS)
@Target(ElementType.TYPE)
public @interface Experimental {
}
