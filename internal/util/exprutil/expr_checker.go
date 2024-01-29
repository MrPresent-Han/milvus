package exprutil

import (
	"math"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus/internal/proto/planpb"
)

type KeyType int64

const (
	PartitionKey  KeyType = iota
	ClusteringKey KeyType = PartitionKey + 1
)

func ParseExprFromPlan(plan *planpb.PlanNode) (*planpb.Expr, error) {
	node := plan.GetNode()

	if node == nil {
		return nil, errors.New("can't get expr from empty plan node")
	}

	var expr *planpb.Expr
	switch node := node.(type) {
	case *planpb.PlanNode_VectorAnns:
		expr = node.VectorAnns.GetPredicates()
	case *planpb.PlanNode_Query:
		expr = node.Query.GetPredicates()
	default:
		return nil, errors.New("unsupported plan node type")
	}

	return expr, nil
}

func ParsePartitionKeysFromBinaryExpr(expr *planpb.BinaryExpr, keyType KeyType) ([]*planpb.GenericValue, bool) {
	leftRes, leftInRange := ParseKeysFromExpr(expr.Left, keyType)
	rightRes, rightInRange := ParseKeysFromExpr(expr.Right, keyType)

	if expr.Op == planpb.BinaryExpr_LogicalAnd {
		// case: partition_key_field in [7, 8] && partition_key > 8
		if len(leftRes)+len(rightRes) > 0 {
			leftRes = append(leftRes, rightRes...)
			return leftRes, false
		}

		// case: other_field > 10 && partition_key_field > 8
		return nil, leftInRange || rightInRange
	}

	if expr.Op == planpb.BinaryExpr_LogicalOr {
		// case: partition_key_field in [7, 8] or partition_key > 8
		if leftInRange || rightInRange {
			return nil, true
		}

		// case: partition_key_field in [7, 8] or other_field > 10
		leftRes = append(leftRes, rightRes...)
		return leftRes, false
	}

	return nil, false
}

func ParsePartitionKeysFromUnaryExpr(expr *planpb.UnaryExpr, keyType KeyType) ([]*planpb.GenericValue, bool) {
	res, partitionInRange := ParseKeysFromExpr(expr.GetChild(), keyType)
	if expr.Op == planpb.UnaryExpr_Not {
		// case: partition_key_field not in [7, 8]
		if len(res) != 0 {
			return nil, true
		}

		// case: other_field not in [10]
		return nil, partitionInRange
	}

	// UnaryOp only includes "Not" for now
	return res, partitionInRange
}

func ParsePartitionKeysFromTermExpr(expr *planpb.TermExpr, keyType KeyType) ([]*planpb.GenericValue, bool) {
	if keyType == PartitionKey && expr.GetColumnInfo().GetIsPartitionKey() {
		return expr.GetValues(), false
	} else if keyType == ClusteringKey && expr.GetColumnInfo().GetIsClusteringKey() {
		return expr.GetValues(), false
	}
	return nil, false
}

func ParsePartitionKeysFromUnaryRangeExpr(expr *planpb.UnaryRangeExpr, keyType KeyType) ([]*planpb.GenericValue, bool) {
	if expr.GetOp() == planpb.OpType_Equal {
		if expr.GetColumnInfo().GetIsPartitionKey() && keyType == PartitionKey ||
			expr.GetColumnInfo().GetIsClusteringKey() && keyType == ClusteringKey {
			return []*planpb.GenericValue{expr.Value}, false
		}
	}
	return nil, true
}

func ParseKeysFromExpr(expr *planpb.Expr, keyType KeyType) ([]*planpb.GenericValue, bool) {
	var res []*planpb.GenericValue
	keyInRange := false
	switch expr := expr.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		res, keyInRange = ParsePartitionKeysFromBinaryExpr(expr.BinaryExpr, keyType)
	case *planpb.Expr_UnaryExpr:
		res, keyInRange = ParsePartitionKeysFromUnaryExpr(expr.UnaryExpr, keyType)
	case *planpb.Expr_TermExpr:
		res, keyInRange = ParsePartitionKeysFromTermExpr(expr.TermExpr, keyType)
	case *planpb.Expr_UnaryRangeExpr:
		res, keyInRange = ParsePartitionKeysFromUnaryRangeExpr(expr.UnaryRangeExpr, keyType)
	}

	return res, keyInRange
}

func ParseKeys(expr *planpb.Expr, kType KeyType) []*planpb.GenericValue {
	res, keyInRange := ParseKeysFromExpr(expr, kType)
	if keyInRange {
		res = nil
	}

	return res
}

type PlanRange struct {
	left         *planpb.GenericValue
	right        *planpb.GenericValue
	includeLeft  bool
	includeRight bool
}

func (planRange *PlanRange) ToIntRange() *IntRange {
	iRange := &IntRange{}
	if planRange.left == nil {
		iRange.left = math.MinInt64
		iRange.includeLeft = false
	} else {
		iRange.left = planRange.left.GetInt64Val()
		iRange.includeLeft = planRange.includeLeft
	}

	if planRange.right == nil {
		iRange.right = math.MaxInt64
		iRange.includeRight = false
	} else {
		iRange.right = planRange.right.GetInt64Val()
		iRange.includeRight = planRange.includeRight
	}
	return iRange
}

func (planRange *PlanRange) ToStrRange() *StrRange {
	sRange := &StrRange{}
	if planRange.left == nil {
		sRange.left = ""
		sRange.includeLeft = false
	} else {
		sRange.left = planRange.left.GetStringVal()
		sRange.includeLeft = planRange.includeLeft
	}

	if planRange.right == nil {
		sRange.right = ""
		sRange.includeRight = false
	} else {
		sRange.right = planRange.right.GetStringVal()
		sRange.includeRight = planRange.includeRight
	}
	return sRange
}

type IntRange struct {
	left         int64
	right        int64
	includeLeft  bool
	includeRight bool
}

func NewIntRange(l int64, r int64, includeL bool, includeR bool) *IntRange {
	return &IntRange{
		left:         l,
		right:        r,
		includeLeft:  includeL,
		includeRight: includeR,
	}
}

func IntRangeOverlap(range1 *IntRange, range2 *IntRange) bool {
	var leftBound int64
	if range1.left < range2.left {
		leftBound = range2.left
	} else {
		leftBound = range1.left
	}
	var rightBound int64
	if range1.right < range2.right {
		rightBound = range1.right
	} else {
		rightBound = range2.right
	}
	return leftBound <= rightBound
}

type StrRange struct {
	left         string
	right        string
	includeLeft  bool
	includeRight bool
}

func NewStrRange(l string, r string, includeL bool, includeR bool) *StrRange {
	return &StrRange{
		left:         l,
		right:        r,
		includeLeft:  includeL,
		includeRight: includeR,
	}
}

func StrRangeOverlap(range1 *StrRange, range2 *StrRange) bool {
	var leftBound string
	if range1.left < range2.left {
		leftBound = range2.left
	} else {
		leftBound = range1.left
	}
	var rightBound string
	if range1.right < range2.right || range2.right == "" {
		rightBound = range1.right
	} else {
		rightBound = range2.right
	}
	return leftBound <= rightBound
}

func ParseRanges(expr *planpb.Expr, kType KeyType) ([]*PlanRange, bool) {
	var res []*PlanRange
	matchALL := true
	switch expr := expr.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		res, matchALL = ParseRangesFromBinaryExpr(expr.BinaryExpr, kType)
	case *planpb.Expr_UnaryRangeExpr:
		res, matchALL = ParseRangesFromUnaryRangeExpr(expr.UnaryRangeExpr, kType)
	case *planpb.Expr_TermExpr:
		res, matchALL = ParseRangesFromTermExpr(expr.TermExpr, kType)
	case *planpb.Expr_UnaryExpr:
		res, matchALL = nil, true
		//we don't handle NOT operation, just consider as unable_to_parse_range
	}
	return res, matchALL
}

func ParseRangesFromBinaryExpr(expr *planpb.BinaryExpr, kType KeyType) ([]*PlanRange, bool) {
	//TODO, handle binary expr
	return nil, true
}

func ParseRangesFromUnaryRangeExpr(expr *planpb.UnaryRangeExpr, kType KeyType) ([]*PlanRange, bool) {
	if expr.GetColumnInfo().GetIsPartitionKey() && kType == PartitionKey ||
		expr.GetColumnInfo().GetIsClusteringKey() && kType == ClusteringKey {
		switch expr.GetOp() {
		case planpb.OpType_Equal:
			{
				return []*PlanRange{
					{
						left:         expr.Value,
						right:        expr.Value,
						includeLeft:  true,
						includeRight: true,
					},
				}, false
			}
		case planpb.OpType_GreaterThan:
			{
				return []*PlanRange{
					{
						left:         expr.Value,
						right:        nil,
						includeLeft:  false,
						includeRight: false,
					},
				}, false
			}
		case planpb.OpType_GreaterEqual:
			{
				return []*PlanRange{
					{
						left:         expr.Value,
						right:        nil,
						includeLeft:  true,
						includeRight: false,
					},
				}, false
			}
		case planpb.OpType_LessThan:
			{
				return []*PlanRange{
					{
						left:         nil,
						right:        expr.Value,
						includeLeft:  false,
						includeRight: false,
					},
				}, false
			}
		case planpb.OpType_LessEqual:
			{
				return []*PlanRange{
					{
						left:         nil,
						right:        expr.Value,
						includeLeft:  false,
						includeRight: true,
					},
				}, false
			}
		}
	}
	return nil, true
}

func ParseRangesFromTermExpr(expr *planpb.TermExpr, kType KeyType) ([]*PlanRange, bool) {
	if expr.GetColumnInfo().GetIsPartitionKey() && kType == PartitionKey ||
		expr.GetColumnInfo().GetIsClusteringKey() && kType == ClusteringKey {
		res := make([]*PlanRange, len(expr.GetValues()))
		for _, value := range expr.GetValues() {
			res = append(res, &PlanRange{
				left:         value,
				right:        value,
				includeLeft:  true,
				includeRight: true,
			})
		}
		return res, false
	}
	return nil, false
}
