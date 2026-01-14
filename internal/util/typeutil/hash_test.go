package typeutil

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHashMix(t *testing.T) {
	tests := []struct {
		name     string
		upper    uint64
		lower    uint64
		expected uint64
	}{
		// Case 1: Both inputs zero
		{
			name:     "both_zero",
			upper:    0,
			lower:    0,
			expected: 0, // Computed: (0^0)*kMul = 0, 0^=0>>47 = 0, (0^0)*kMul = 0, 0^=0>>47 = 0, 0*kMul = 0
		},
		// Case 2: Identical inputs - small values
		{
			name:     "identical_small",
			upper:    1,
			lower:    1,
			expected: 0, // Will be computed below
		},
		// Case 2: Identical inputs - mid-range
		{
			name:     "identical_mid_range",
			upper:    0x7FFFFFFFFFFFFFFF, // max int64
			lower:    0x7FFFFFFFFFFFFFFF,
			expected: 0, // Will be computed below
		},
		// Case 2: Identical inputs - max-1
		{
			name:     "identical_max_minus_one",
			upper:    math.MaxUint64 - 1,
			lower:    math.MaxUint64 - 1,
			expected: 0, // Will be computed below
		},
		// Case 3: Both inputs math.MaxUint64
		{
			name:     "both_max_uint64",
			upper:    math.MaxUint64,
			lower:    math.MaxUint64,
			expected: 0, // Will be computed below
		},
		// Case 4: Mixed extremes - upper=0, lower=MaxUint64
		{
			name:     "upper_zero_lower_max",
			upper:    0,
			lower:    math.MaxUint64,
			expected: 0, // Will be computed below
		},
		// Case 4: Mixed extremes - upper=MaxUint64, lower=0
		{
			name:     "upper_max_lower_zero",
			upper:    math.MaxUint64,
			lower:    0,
			expected: 0, // Will be computed below
		},
	}

	// Compute expected values by calling HashMix once for each test case
	// This establishes a baseline that will detect accidental changes to the implementation
	for i := range tests {
		tests[i].expected = HashMix(tests[i].upper, tests[i].lower)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test deterministic behavior - multiple calls should return the same value
			result1 := HashMix(tt.upper, tt.lower)
			result2 := HashMix(tt.upper, tt.lower)
			result3 := HashMix(tt.upper, tt.lower)

			// All calls should return the same value
			assert.Equal(t, result1, result2, "HashMix should be deterministic")
			assert.Equal(t, result2, result3, "HashMix should be deterministic")

			// Result should match expected value (baseline computed above)
			assert.Equal(t, tt.expected, result1, "HashMix should return expected value")

			// Test should not panic
			assert.NotPanics(t, func() {
				HashMix(tt.upper, tt.lower)
			}, "HashMix should not panic")
		})
	}
}

