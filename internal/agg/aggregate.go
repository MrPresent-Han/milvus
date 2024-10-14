package agg

import (
	"fmt"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"regexp"
	"strings"
)

const (
	kSum   = "sum"
	kCount = "count"
	kAvg   = "avg"
	kMin   = "min"
	kMax   = "max"
)

var (
	// Define the regular expression pattern once to avoid repeated concatenation.
	aggregationTypes   = kSum + `|` + kCount + `|` + kAvg + `|` + kMin + `|` + kMax
	aggregationPattern = regexp.MustCompile(`(?i)^(` + aggregationTypes + `)\s*\(\s*([\w\*]*)\s*\)$`)
)

// MatchAggregationExpression return isAgg, operator name, operator parameter
func MatchAggregationExpression(expression string) (bool, string, string) {
	// FindStringSubmatch returns the full match and submatches.
	matches := aggregationPattern.FindStringSubmatch(expression)
	if len(matches) > 0 {
		// Return true, the operator, and the captured parameter.
		return true, strings.ToLower(matches[1]), strings.TrimSpace(matches[2])
	}
	return false, "", ""
}

func NewAggregate(aggregateName string, aggFieldID int64) (AggregateBase, error) {
	switch aggregateName {
	case kCount:
		return &CountAggregate{}, nil
	case kSum:
		return &SumAggregate{}, nil
	case kAvg:
		return &AverageAggregate{}, nil
	case kMin:
		return &MinAggregate{}, nil
	case kMax:
		return &MaxAggregate{}, nil
	default:
		return nil, merr.WrapErrParameterInvalid(aggregationTypes, fmt.Sprintf("invalid Aggregation operator %s", aggregateName))
	}
}

type AggregateBase struct {
	fieldID int64
}

func (aggBase *AggregateBase) aggFieldID() int64 {
	return aggBase.fieldID
}
func (aggBase *AggregateBase) name() string {
	return ""
}
func (aggBase *AggregateBase) compute() {
}

type SumAggregate struct {
	AggregateBase
}

func (avg *SumAggregate) name() string {
	return kSum
}
func (avg *SumAggregate) compute() {

}

type CountAggregate struct {
	AggregateBase
}

func (avg *CountAggregate) name() string {
	return kCount
}
func (avg *CountAggregate) compute() {

}

type AverageAggregate struct {
	AggregateBase
}

func (avg *AverageAggregate) name() string {
	return kAvg
}
func (avg *AverageAggregate) compute() {

}

type MinAggregate struct {
	AggregateBase
}

func (avg *MinAggregate) name() string {
	return kMin
}
func (avg *MinAggregate) compute() {

}

type MaxAggregate struct {
	AggregateBase
}

func (avg *MaxAggregate) name() string {
	return kMax
}

func (avg *MaxAggregate) compute() {

}
