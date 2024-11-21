package segments

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus/internal/agg"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	typeutil2 "github.com/milvus-io/milvus/internal/util/typeutil"

)

type AggReducer struct {
	groupByFieldIds []int64
	aggregates      []*planpb.Aggregate
	schema          *schemapb.CollectionSchema
	hashValsMap     map[uint64]*agg.Bucket
	hashes          []uint64
}

func CalculateRowNumber(results []*segcorepb.RetrieveResults) int64 {
	rowNumber := 0
	for _, result := range results {
		fieldsData := result.GetFieldsData()
		if len(fieldsData) < 1 {
			return 0
		}
		firstFieldData := fieldsData[0]
		switch firstFieldData.GetType() {
		case schemapb.DataType_Bool:
			rowNumber += len(firstFieldData.GetScalars().GetBoolData().Data)
		case schemapb.DataType_Int8:
		case schemapb.DataType_Int16:
		case schemapb.DataType_Int32:
			rowNumber += len(firstFieldData.GetScalars().GetIntData().Data)
		case schemapb.DataType_Int64:
			rowNumber += len(firstFieldData.GetScalars().GetLongData().Data)
		case schemapb.DataType_VarChar:
		case schemapb.DataType_String:
			rowNumber += len(firstFieldData.GetScalars().GetStringData().Data)
		case schemapb.DataType_Float:
			rowNumber += len(firstFieldData.GetScalars().GetFloatData().Data)
		case schemapb.DataType_Double:
			rowNumber += len(firstFieldData.GetScalars().GetDoubleData().Data)
		}
	}
	return int64(rowNumber)
}

func (reducer *AggReducer) Reduce(ctx context.Context, results []*segcorepb.RetrieveResults, segments []Segment, plan *RetrievePlan) (*segcorepb.RetrieveResults, error) {
	if results == nil || len(results) == 0 {
		return nil, fmt.Errorf("no input segment's retrieved results can be reduced")
	}
	//0. set up aggregates
	aggs := make([]agg.AggregateBase, len(reducer.aggregates))
	for idx, aggPb := range reducer.aggregates {
		agg, err := agg.FromPB(aggPb)
		if err != nil {
			return nil, err
		}
		aggs[idx] = agg
	}

	//1. set up hashers and accumulators
	rowNumber := CalculateRowNumber(results)
	reducer.hashes = make([]uint64, rowNumber)
	numGroupingKeys := len(reducer.groupByFieldIds)
	hashers := make([]agg.FieldAccessor, 0, numGroupingKeys)
	accumulators := make([]agg.FieldAccessor, 0)
	firstFieldData := results[0].GetFieldsData()
	for idx, fieldData := range firstFieldData {
		if idx < numGroupingKeys {
			hasher, err := agg.NewFieldAccessor(fieldData.GetType())
			if err != nil {
				return nil, err
			}
			hashers = append(hashers, hasher)
		}
		if idx >= numGroupingKeys {
			accumulator, err := agg.NewFieldAccessor(fieldData.GetType())
			if err != nil {
				return nil, err
			}
			accumulators = append(accumulators, accumulator)
		}
	}

	// 2. compute hash values for all rows in the result retrieved
	outputColumnCount := -1
	rowIdx := 0
	for _, result := range results {
		fieldDatas := result.GetFieldsData()
		if outputColumnCount == -1 {
			outputColumnCount = len(fieldDatas)
		} else if outputColumnCount != len(fieldDatas) {
			return nil, fmt.Errorf("retrieved results from different segments have different size of columns")
		}
		if outputColumnCount == 0 {
			return nil, fmt.Errorf("retrieved results have no column data")
		}
		rowCount := -1
		for i := 0; i < outputColumnCount; i++ {
			fieldData := fieldDatas[i]
			if i < numGroupingKeys {
				hashers[i].SetVals(fieldData)
			} else {
				accumulators[i-numGroupingKeys].SetVals(fieldData)
			}
			if rowCount == -1 {
				rowCount = hashers[i].RowCount()
			} else if rowCount != hashers[i].RowCount() {
				return nil, fmt.Errorf("field data:%d for different columns have different row count, %d vs %d, wrong state",
					i, rowCount, hashers[i].RowCount())
			}
		}
		for row := 0; row < rowCount; row++ {
			rowEntries := make([]*agg.Entry, outputColumnCount)
			for col := 0; col < outputColumnCount; col++ {
				if col < numGroupingKeys {
					if col > 0 {
						reducer.hashes[rowIdx] = typeutil2.HashMix(reducer.hashes[rowIdx], hashers[col].Hash(row))
					} else {
						reducer.hashes[rowIdx] = hashers[col].Hash(row)
					}
					rowEntries[col] = agg.NewEntry(hashers[col].ValAt(row))
				} else {
					rowEntries[col] = agg.NewEntry(accumulators[col-numGroupingKeys].ValAt(row))
				}
			}
			newRow := agg.NewRow(rowEntries)
			if bucket := reducer.hashValsMap[reducer.hashes[rowIdx]]; bucket == nil {
				newBucket := agg.NewBucket()
				newBucket.AddRow(newRow)
				reducer.hashValsMap[reducer.hashes[rowIdx]] = newBucket
			} else {
				if rowIdx := bucket.Find(newRow, numGroupingKeys); rowIdx == agg.NONE {
					bucket.AddRow(newRow)
				} else {
					bucket.Accumulate(newRow, rowIdx, numGroupingKeys, )
				}
			}
			rowIdx++
		}
	}

	return nil, nil
}

}
