package segments

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"testing"
)

type AggReduceSuite struct {
	suite.Suite
}

func TestSegCoreAggReduce(t *testing.T) {
	groupByFieldIds := make([]int64, 1)
	groupByFieldIds[0] = 101
	aggregates := make([]*planpb.Aggregate, 1)
	aggregates[0] = &planpb.Aggregate{
		Op:      planpb.AggregateOp_sum,
		FieldId: 102,
	}

	aggReducer := NewSegcoreAggReducer(groupByFieldIds, aggregates)
	results := make([]*segcorepb.RetrieveResults, 2)
	{
		fieldData1 := &schemapb.FieldData{
			Type: schemapb.DataType_Int16,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 4, 8, 11},
						},
					},
				},
			},
		}
		fieldData2 := &schemapb.FieldData{
			Type: schemapb.DataType_Int64,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{12, 33, 24, 48, 11},
						},
					},
				},
			},
		}
		results[0] = &segcorepb.RetrieveResults{
			FieldsData: []*schemapb.FieldData{fieldData1, fieldData2},
		}
	}
	{
		fieldData1 := &schemapb.FieldData{
			Type: schemapb.DataType_Int16,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 5, 9, 11},
						},
					},
				},
			},
		}
		fieldData2 := &schemapb.FieldData{
			Type: schemapb.DataType_Int64,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{12, 33, 15, 18, 11},
						},
					},
				},
			},
		}
		results[1] = &segcorepb.RetrieveResults{
			FieldsData: []*schemapb.FieldData{fieldData1, fieldData2},
		}
	}

	reducedRes, err := aggReducer.Reduce(context.Background(), results, nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, reducedRes)

	resInt32 := []int32{2, 3, 4, 5, 8, 9, 11}
	resInt64 := []int64{24, 66, 24, 15, 48, 18, 22}
	assert.EqualValues(t, resInt32, reducedRes.GetFieldsData()[0])
	assert.EqualValues(t, resInt64, reducedRes.GetFieldsData()[1])
}

func TestSegCoreAggReduceError(t *testing.T) {
	groupByFieldIds := make([]int64, 1)
	groupByFieldIds[0] = 101
	aggregates := make([]*planpb.Aggregate, 1)
	aggregates[0] = &planpb.Aggregate{
		Op:      planpb.AggregateOp_sum,
		FieldId: 102,
	}

	aggReducer := NewSegcoreAggReducer(groupByFieldIds, aggregates)
	results := make([]*segcorepb.RetrieveResults, 2)
	// should report error when
	// field data's lengths are different
	{
		fieldData1 := &schemapb.FieldData{
			Type: schemapb.DataType_Int16,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 4, 8, 11},
						},
					},
				},
			},
		}
		fieldData2 := &schemapb.FieldData{
			Type: schemapb.DataType_Int64,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{12, 33},
						},
					},
				},
			},
		}
		results[0] = &segcorepb.RetrieveResults{
			FieldsData: []*schemapb.FieldData{fieldData1, fieldData2},
		}
	}
	{
		fieldData1 := &schemapb.FieldData{
			Type: schemapb.DataType_Int16,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_IntData{
						IntData: &schemapb.IntArray{
							Data: []int32{2, 3, 5, 9, 11},
						},
					},
				},
			},
		}
		fieldData2 := &schemapb.FieldData{
			Type: schemapb.DataType_Int64,
			Field: &schemapb.FieldData_Scalars{
				Scalars: &schemapb.ScalarField{
					Data: &schemapb.ScalarField_LongData{
						LongData: &schemapb.LongArray{
							Data: []int64{12},
						},
					},
				},
			},
		}
		results[1] = &segcorepb.RetrieveResults{
			FieldsData: []*schemapb.FieldData{fieldData1, fieldData2},
		}
	}

	reducedRes, err := aggReducer.Reduce(context.Background(), results, nil, nil)
	assert.Error(t, err)
	assert.Nil(t, reducedRes)
}

func TestAggReduce(t *testing.T) {
	paramtable.Init()
	suite.Run(t, new(AggReduceSuite))
}
