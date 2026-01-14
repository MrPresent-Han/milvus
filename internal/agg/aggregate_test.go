package agg

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestMatchAggregationExpression(t *testing.T) {
	tests := []struct {
		expression       string
		expectedIsValid  bool
		expectedOperator string
		expectedParam    string
	}{
		// Basic valid expressions
		{"count(*)", true, "count", "*"},
		{"count(a)", true, "count", "a"},
		{"sum(b)", true, "sum", "b"},
		{"avg(c)", true, "avg", "c"},
		{"min(d)", true, "min", "d"},
		{"max(e)", true, "max", "e"},

		// Case insensitive operators
		{"COUNT(*)", true, "count", "*"},
		{"SUM(x)", true, "sum", "x"},
		{"AVG(y)", true, "avg", "y"},
		{"MIN(z)", true, "min", "z"},
		{"MAX(w)", true, "max", "w"},
		{"CoUnT(*)", true, "count", "*"},
		{"SuM(a)", true, "sum", "a"},

		// With spaces
		{"sum ( x )", true, "sum", "x"},
		{"AVG( y )", true, "avg", "y"},
		{"count ( * )", true, "count", "*"},
		{"min  (  field_name  )", true, "min", "field_name"},
		{"max( field123 )", true, "max", "field123"},

		// Without spaces
		{"sum(x)", true, "sum", "x"},
		{"avg(y)", true, "avg", "y"},
		{"count(*)", true, "count", "*"},

		// Field names with underscores and numbers
		{"sum(field_1)", true, "sum", "field_1"},
		{"count(field_name_123)", true, "count", "field_name_123"},
		{"avg(field123)", true, "avg", "field123"},
		{"min(f1)", true, "min", "f1"},
		{"max(f_1)", true, "max", "f_1"},

		// Field names with mixed case
		{"sum(FieldName)", true, "sum", "FieldName"},
		{"count(FIELD_NAME)", true, "count", "FIELD_NAME"},
		{"avg(fieldName)", true, "avg", "fieldName"},

		// Long field names
		{"sum(very_long_field_name_that_might_be_used_in_real_world)", true, "sum", "very_long_field_name_that_might_be_used_in_real_world"},
		{"count(another_very_long_field_name_with_numbers_123456)", true, "count", "another_very_long_field_name_with_numbers_123456"},

		// Edge case: empty parameter (regex allows empty string)
		{"count()", true, "count", ""},
		{"sum()", true, "sum", ""},
		{"avg()", true, "avg", ""},

		// Invalid expressions
		{"invalidExpression", false, "", ""},
		{"count", false, "", ""},
		{"count(", false, "", ""},
		{"count)", false, "", ""},
		{"(count)", false, "", ""},
		{"count(*", false, "", ""},
		{"count*)", false, "", ""},
		{"sum", false, "", ""},
		{"avg", false, "", ""},
		{"min", false, "", ""},
		{"max", false, "", ""},
		{"", false, "", ""},
		{"count(*))", false, "", ""},
		{"count((*)", false, "", ""},
		{"count(* )", true, "count", "*"},
		{"count( *)", true, "count", "*"},

		// Invalid operators
		{"invalid(*)", false, "", ""},
		{"summation(*)", false, "", ""},
		{"counts(*)", false, "", ""},
		{"average(*)", false, "", ""},
		{"minimum(*)", false, "", ""},
		{"maximum(*)", false, "", ""},

		// Edge cases with special characters in field names (should fail based on regex)
		{"sum(field-name)", false, "", ""},   // hyphen not allowed
		{"count(field.name)", false, "", ""}, // dot not allowed
		{"avg(field name)", false, "", ""},   // space not allowed
		{"min(field@name)", false, "", ""},   // @ not allowed

		// Multiple parentheses (should fail)
		{"count((*))", false, "", ""},
		{"sum((x))", false, "", ""},

		// Extra characters
		{"count(*)extra", false, "", ""},
		{"sum(x)extra", false, "", ""},
		{"extra count(*)", false, "", ""},
		{"count(*) extra", false, "", ""},

		// Whitespace variations
		{"  count(*)  ", false, "", ""}, // leading/trailing spaces not allowed
		{"count  (*)", true, "count", "*"},
		{"count(  *)", true, "count", "*"},
		{"count(*  )", true, "count", "*"},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("expr_%q", test.expression), func(t *testing.T) {
		isValid, operator, param := MatchAggregationExpression(test.expression)
		if isValid != test.expectedIsValid || operator != test.expectedOperator || param != test.expectedParam {
			t.Errorf("MatchAggregationExpression(%q) = (%v, %q, %q), want (%v, %q, %q)",
				test.expression, isValid, operator, param, test.expectedIsValid, test.expectedOperator, test.expectedParam)
		}
		})
	}
}

// generateTestExpressions generates a large set of test expressions for benchmarking
func generateTestExpressions() []string {
	operators := []string{"count", "sum", "avg", "min", "max"}
	fieldNames := []string{
		"*",
		"field1",
		"field_name",
		"FieldName",
		"FIELD_NAME",
		"field123",
		"f1",
		"very_long_field_name_that_might_be_used_in_real_world",
		"another_very_long_field_name_with_numbers_123456",
		"x",
		"y",
		"z",
		"a",
		"b",
		"c",
		"d",
		"e",
		"f",
		"g",
		"h",
		"i",
		"j",
		"k",
		"l",
		"m",
		"n",
		"o",
		"p",
		"q",
		"r",
		"s",
		"t",
		"u",
		"v",
		"w",
	}

	expressions := make([]string, 0, len(operators)*len(fieldNames)*4) // 4 variations per combination

	// Generate valid expressions
	for _, op := range operators {
		for _, field := range fieldNames {
			// Variation 1: no spaces
			expressions = append(expressions, fmt.Sprintf("%s(%s)", op, field))
			// Variation 2: spaces around parentheses
			expressions = append(expressions, fmt.Sprintf("%s ( %s )", op, field))
			// Variation 3: uppercase operator
			expressions = append(expressions, fmt.Sprintf("%s(%s)", strings.ToUpper(op), field))
			// Variation 4: mixed case operator
			expressions = append(expressions, fmt.Sprintf("%s(%s)", strings.Title(op), field))
		}
	}

	// Add some invalid expressions to make it more realistic
	invalidExpressions := []string{
		"invalid(*)",
		"count",
		"sum",
		"avg",
		"min",
		"max",
		"count()",
		"sum(",
		"avg)",
		"count(*))",
		"extra count(*)",
		"count(*) extra",
		"",
		"not_an_aggregation",
		"count((*))",
		"sum(field-name)",
		"avg(field.name)",
	}

	expressions = append(expressions, invalidExpressions...)

	return expressions
}

// BenchmarkMatchAggregationExpression benchmarks the function with a large set of expressions
func BenchmarkMatchAggregationExpression(b *testing.B) {
	expressions := generateTestExpressions()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, expr := range expressions {
			MatchAggregationExpression(expr)
		}
	}
}

// BenchmarkMatchAggregationExpression_Single benchmarks a single expression
func BenchmarkMatchAggregationExpression_Single(b *testing.B) {
	expr := "count(field_name)"
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		MatchAggregationExpression(expr)
	}
}

// BenchmarkMatchAggregationExpression_ValidOnly benchmarks only valid expressions
func BenchmarkMatchAggregationExpression_ValidOnly(b *testing.B) {
	validExpressions := []string{
		"count(*)",
		"sum(field1)",
		"avg(field_name)",
		"min(FieldName)",
		"max(FIELD_NAME)",
		"COUNT(*)",
		"SUM(field1)",
		"AVG(field_name)",
		"MIN(FieldName)",
		"MAX(FIELD_NAME)",
		"count ( * )",
		"sum ( field1 )",
		"avg ( field_name )",
		"min ( FieldName )",
		"max ( FIELD_NAME )",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, expr := range validExpressions {
			MatchAggregationExpression(expr)
		}
	}
}

// BenchmarkMatchAggregationExpression_InvalidOnly benchmarks only invalid expressions
func BenchmarkMatchAggregationExpression_InvalidOnly(b *testing.B) {
	invalidExpressions := []string{
		"invalid(*)",
		"count",
		"sum",
		"avg",
		"min",
		"max",
		"count()",
		"sum(",
		"avg)",
		"count(*))",
		"extra count(*)",
		"count(*) extra",
		"",
		"not_an_aggregation",
		"count((*))",
		"sum(field-name)",
		"avg(field.name)",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, expr := range invalidExpressions {
			MatchAggregationExpression(expr)
		}
	}
}

// BenchmarkMatchAggregationExpression_LongFieldNames benchmarks with long field names
func BenchmarkMatchAggregationExpression_LongFieldNames(b *testing.B) {
	longFieldNames := []string{
		"very_long_field_name_that_might_be_used_in_real_world",
		"another_very_long_field_name_with_numbers_123456",
		"field_name_with_many_underscores_and_numbers_123456789",
		"a_very_very_very_long_field_name_that_exceeds_normal_length",
	}

	expressions := make([]string, 0, len(longFieldNames)*5)
	for _, field := range longFieldNames {
		for _, op := range []string{"count", "sum", "avg", "min", "max"} {
			expressions = append(expressions, fmt.Sprintf("%s(%s)", op, field))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, expr := range expressions {
			MatchAggregationExpression(expr)
		}
	}
}

// BenchmarkMatchAggregationExpression_CaseVariations benchmarks case variations
func BenchmarkMatchAggregationExpression_CaseVariations(b *testing.B) {
	baseExpr := "count(field_name)"
	expressions := []string{
		baseExpr,
		"COUNT(field_name)",
		"Count(field_name)",
		"CoUnT(field_name)",
		"COUNT(FIELD_NAME)",
		"Count(FieldName)",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, expr := range expressions {
			MatchAggregationExpression(expr)
		}
	}
}

// TestMatchAggregationExpression_100KTimes tests 100,000 matches and reports the time
func TestMatchAggregationExpression_100KTimes(t *testing.T) {
	const iterations = 100000
	expressions := generateTestExpressions()

	// Warm up
	for i := 0; i < 1000; i++ {
		for _, expr := range expressions {
			MatchAggregationExpression(expr)
		}
	}

	// Actual test
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for _, expr := range expressions {
			MatchAggregationExpression(expr)
		}
	}
	elapsed := time.Since(start)

	totalMatches := iterations * len(expressions)
	avgTimePerMatch := elapsed / time.Duration(totalMatches)

	t.Logf("=== Performance Statistics for 100K Matches ===")
	t.Logf("Total matches: %d", totalMatches)
	t.Logf("Total elapsed time: %v", elapsed)
	t.Logf("Average time per match: %v", avgTimePerMatch)
	t.Logf("Matches per second: %.2f", float64(totalMatches)/elapsed.Seconds())
	t.Logf("Matches per millisecond: %.2f", float64(totalMatches)/float64(elapsed.Milliseconds()))
}

// TestMatchAggregationExpression_100KTimes_SingleExpr tests 100,000 matches with a single expression
func TestMatchAggregationExpression_100KTimes_SingleExpr(t *testing.T) {
	const iterations = 100000
	expr := "count(field_name)"

	// Warm up
	for i := 0; i < 1000; i++ {
		MatchAggregationExpression(expr)
	}

	// Actual test
	start := time.Now()
	for i := 0; i < iterations; i++ {
		MatchAggregationExpression(expr)
	}
	elapsed := time.Since(start)

	avgTimePerMatch := elapsed / time.Duration(iterations)

	t.Logf("=== Performance Statistics for 100K Matches (Single Expression) ===")
	t.Logf("Expression: %q", expr)
	t.Logf("Total matches: %d", iterations)
	t.Logf("Total elapsed time: %v", elapsed)
	t.Logf("Average time per match: %v", avgTimePerMatch)
	t.Logf("Matches per second: %.2f", float64(iterations)/elapsed.Seconds())
	t.Logf("Matches per millisecond: %.2f", float64(iterations)/float64(elapsed.Milliseconds()))
}

// TestMatchAggregationExpression_100KTimes_Mixed tests 100,000 matches with mixed valid/invalid expressions
func TestMatchAggregationExpression_100KTimes_Mixed(t *testing.T) {
	const iterations = 100000
	mixedExpressions := []string{
		"count(*)",
		"sum(field1)",
		"avg(field_name)",
		"min(FieldName)",
		"max(FIELD_NAME)",
		"COUNT(*)",
		"SUM(field1)",
		"AVG(field_name)",
		"invalid(*)",
		"count",
		"sum",
		"avg",
		"count()",
		"sum()",
		"count(*))",
		"extra count(*)",
		"",
	}

	// Warm up
	for i := 0; i < 1000; i++ {
		for _, expr := range mixedExpressions {
			MatchAggregationExpression(expr)
		}
	}

	// Actual test
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for _, expr := range mixedExpressions {
			MatchAggregationExpression(expr)
		}
	}
	elapsed := time.Since(start)

	totalMatches := iterations * len(mixedExpressions)
	avgTimePerMatch := elapsed / time.Duration(totalMatches)

	t.Logf("=== Performance Statistics for 100K Matches (Mixed Expressions) ===")
	t.Logf("Number of expressions: %d", len(mixedExpressions))
	t.Logf("Total matches: %d", totalMatches)
	t.Logf("Total elapsed time: %v", elapsed)
	t.Logf("Average time per match: %v", avgTimePerMatch)
	t.Logf("Matches per second: %.2f", float64(totalMatches)/elapsed.Seconds())
	t.Logf("Matches per millisecond: %.2f", float64(totalMatches)/float64(elapsed.Milliseconds()))
}
