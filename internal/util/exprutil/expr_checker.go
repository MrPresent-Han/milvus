package exprutil

import (
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
	RightRes, rightInRange := ParseKeysFromExpr(expr.Right, keyType)

	if expr.Op == planpb.BinaryExpr_LogicalAnd {
		// case: partition_key_field in [7, 8] && partition_key > 8
		if len(leftRes)+len(RightRes) > 0 {
			leftRes = append(leftRes, RightRes...)
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
		leftRes = append(leftRes, RightRes...)
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
	partitionKeyInRange := false
	switch expr := expr.GetExpr().(type) {
	case *planpb.Expr_BinaryExpr:
		res, partitionKeyInRange = ParsePartitionKeysFromBinaryExpr(expr.BinaryExpr, keyType)
	case *planpb.Expr_UnaryExpr:
		res, partitionKeyInRange = ParsePartitionKeysFromUnaryExpr(expr.UnaryExpr, keyType)
	case *planpb.Expr_TermExpr:
		res, partitionKeyInRange = ParsePartitionKeysFromTermExpr(expr.TermExpr, keyType)
	case *planpb.Expr_UnaryRangeExpr:
		res, partitionKeyInRange = ParsePartitionKeysFromUnaryRangeExpr(expr.UnaryRangeExpr, keyType)
	}

	return res, partitionKeyInRange
}

func ParseKeys(expr *planpb.Expr, kType KeyType) []*planpb.GenericValue {
	res, partitionKeyInRange := ParseKeysFromExpr(expr, kType)
	if partitionKeyInRange {
		res = nil
	}

	return res
}
