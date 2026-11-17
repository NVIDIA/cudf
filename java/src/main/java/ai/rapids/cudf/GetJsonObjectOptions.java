/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Options that control how {@link ColumnView#getJSONObject(Scalar, GetJsonObjectOptions)}
 * parses JSON.
 */
public final class GetJsonObjectOptions {

    public static GetJsonObjectOptions DEFAULT = new GetJsonObjectOptions.Builder().build();

    private final boolean allowSingleQuotes;
    private final boolean stripQuotesFromSingleStrings;
    private final boolean missingFieldsAsNulls;

    private GetJsonObjectOptions(Builder builder) {
        this.allowSingleQuotes = builder.allowSingleQuotes;
        this.stripQuotesFromSingleStrings = builder.stripQuotesFromSingleStrings;
        this.missingFieldsAsNulls = builder.missingFieldsAsNulls;
    }

    public boolean isAllowSingleQuotes() {
        return allowSingleQuotes;
    }

    public boolean isStripQuotesFromSingleStrings() {
        return stripQuotesFromSingleStrings;
    }

    public boolean isMissingFieldsAsNulls() {
        return missingFieldsAsNulls;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private boolean allowSingleQuotes = false;
        private boolean stripQuotesFromSingleStrings = true;
        private boolean missingFieldsAsNulls = false;

        public Builder allowSingleQuotes(boolean allowSingleQuotes) {
            this.allowSingleQuotes = allowSingleQuotes;
            return this;
        }

        public Builder stripQuotesFromSingleStrings(boolean stripQuotesFromSingleStrings) {
            this.stripQuotesFromSingleStrings = stripQuotesFromSingleStrings;
            return this;
        }

        public Builder missingFieldsAsNulls(boolean missingFieldsAsNulls) {
            this.missingFieldsAsNulls = missingFieldsAsNulls;
            return this;
        }

        public GetJsonObjectOptions build() {
            return new GetJsonObjectOptions(this);
        }
    }
}
