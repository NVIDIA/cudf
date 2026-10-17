/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * The side of a string to which padding characters are added.
 * Used by {@link ColumnView#pad(int, PadSide, String)}.
 */
public enum PadSide {
    LEFT(0),
    RIGHT(1),
    BOTH(2);

    private static final PadSide[] PAD_SIDES = PadSide.values();
    final int nativeId;

    PadSide(int nativeId) {
        this.nativeId = nativeId;
    }

    public int getNativeId() {
        return nativeId;
    }

    public static PadSide fromNative(int nativeId) {
        for (PadSide type : PAD_SIDES) {
            if (type.nativeId == nativeId) {
                return type;
            }
        }
        throw new IllegalArgumentException("Could not translate " + nativeId + " into a PadSide");
    }
}
