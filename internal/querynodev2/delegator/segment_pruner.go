package delegator

import (
	"sort"
	"strconv"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus-storage/go/common/log"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/planpb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/clustering"
	"github.com/milvus-io/milvus/internal/util/exprutil"
	"github.com/milvus-io/milvus/pkg/common"
	"github.com/milvus-io/milvus/pkg/util/distance"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
)

const defaultFilterRatio float64 = 0.5

type PruneInfo struct {
	filterRatio float64
}

func PruneSegments(partitionStats map[UniqueID]*storage.PartitionStatsSnapshot,
	searchReq *internalpb.SearchRequest,
	queryReq *internalpb.RetrieveRequest,
	schema *schemapb.CollectionSchema,
	sealedSegments []SnapshotItem,
	info PruneInfo) {

	//1. calculate filtered segments
	filteredSegments := make(map[UniqueID]struct{}, 0)
	if searchReq != nil {
		for _, partID := range searchReq.GetPartitionIDs() {
			partStats := partitionStats[partID]
			for _, field := range schema.Fields {
				FilterSegmentsByVector(partStats, searchReq, field, filteredSegments, info.filterRatio)
			}
		}
	} else if queryReq != nil {
		for _, partID := range queryReq.GetPartitionIDs() {
			partStats := partitionStats[partID]
			for _, field := range schema.Fields {
				FilterSegmentsOnScalarField(partStats, queryReq, field, filteredSegments)
			}
		}
	}

	//2. remove filtered segments from sealed segment list
	if len(filteredSegments) > 0 {
		for idx, item := range sealedSegments {
			newSegments := make([]SegmentEntry, 0)
			for _, segment := range item.Segments {
				if _, ok := filteredSegments[segment.SegmentID]; !ok {
					newSegments = append(newSegments, segment)
				}
			}
			item.Segments = newSegments
			sealedSegments[idx] = item
		}
	}
}

func FilterSegmentsByVector(partitionStats *storage.PartitionStatsSnapshot,
	searchReq *internalpb.SearchRequest,
	keyField *schemapb.FieldSchema,
	filteredSegments map[UniqueID]struct{},
	filterRatio float64) {
	//0. parse searched vectors
	var vectorsHolder commonpb.PlaceholderGroup
	err := proto.Unmarshal(searchReq.GetPlaceholderGroup(), &vectorsHolder)
	if err != nil || len(vectorsHolder.GetPlaceholders()) == 0 {
		return
	}
	vectorsBytes := vectorsHolder.GetPlaceholders()[0].GetValues()

	//1. parse parameters
	type segmentDisStruct struct {
		segmentID UniqueID
		distance  float32
		rows      int //for keep track of sufficiency of topK
	}
	dimStr, err := funcutil.GetAttrByKeyFromRepeatedKV(common.DimKey, keyField.GetTypeParams())
	if err != nil {
		return
	}
	dimValue, err := strconv.ParseInt(dimStr, 10, 32)
	if err != nil {
		return
	} //this parse cost may incur high cost for search-request

	//2. calculate vectors' distances
	neededSegments := make(map[UniqueID]struct{})
	for _, vecBytes := range vectorsBytes {
		segmentsToSearch := make([]segmentDisStruct, 0)
		for segId, segStats := range partitionStats.SegmentStats {
			//if _, needed := neededSegments[segId]; needed {
			//	continue
			//}
			//here, we do not skip needed segments required by former query vector
			//meaning that repeated calculation will be carried and the larger the nq is
			//the more segments have to be included and prune effect will decline
			//1. calculate distances from centroids
			for _, fieldStat := range segStats.FieldStats {
				if fieldStat.FieldID == keyField.GetFieldID() {
					dis, err := clustering.CalcVectorDistance(dimValue, keyField.GetDataType(),
						vecBytes, fieldStat.Centroids[0].GetValue().([]float32), searchReq.GetMetricType())
					//currently, we only support float vector and only one center one segment
					if err != nil {
						neededSegments[segId] = struct{}{}
						//when running across err, we need to set current segment as needed
						//to avoid lose data
						return
					}
					segmentsToSearch = append(segmentsToSearch, segmentDisStruct{
						segmentID: segId,
						distance:  dis[0],
						rows:      segStats.NumRows,
					})
					break
				}
			}
		}
		//2. sort the distances
		switch searchReq.GetMetricType() {
		case distance.L2:
			sort.SliceStable(segmentsToSearch, func(i, j int) bool {
				return segmentsToSearch[i].distance < segmentsToSearch[j].distance
			})
		case distance.IP, distance.COSINE:
			sort.SliceStable(segmentsToSearch, func(i, j int) bool {
				return segmentsToSearch[i].distance > segmentsToSearch[j].distance
			})
		}

		//3. filtered target segments
		segmentCount := len(segmentsToSearch)
		targetSegNum := int(float64(segmentCount) * filterRatio)
		optimizedRowCount := 0
		// set the last n - targetSegNum as being filtered
		for i := 0; i < segmentCount; i++ {
			optimizedRowCount += segmentsToSearch[i].rows
			neededSegments[segmentsToSearch[i].segmentID] = struct{}{}
			if int64(optimizedRowCount) >= searchReq.GetTopk() && i >= targetSegNum {
				break
			}
		}
	}

	//3. set not needed segments as removed
	for segId, _ := range partitionStats.SegmentStats {
		if _, ok := neededSegments[segId]; !ok {
			filteredSegments[segId] = struct{}{}
		}
	}
}

func FilterSegmentsOnScalarField(partitionStats *storage.PartitionStatsSnapshot,
	queryReq *internalpb.RetrieveRequest,
	keyField *schemapb.FieldSchema,
	filteredSegments map[UniqueID]struct{}) {
	//0. parse expr from plan
	plan := planpb.PlanNode{}
	err := proto.Unmarshal(queryReq.GetSerializedExprPlan(), &plan)
	if err != nil {
		log.Error("failed to unmarshall serialized expr from bytes, failed the operation")
		return
	}
	expr, err := exprutil.ParseExprFromPlan(&plan)
	if err != nil {
		log.Error("failed to parse expr from plan, failed the operation")
		return
	}
	partitionVals := exprutil.ParseKeys(expr, exprutil.ClusteringKey)
	if partitionVals == nil {
		return
	}
	//1. try to filter segments
	outRange := func(min storage.ScalarFieldValue, max storage.ScalarFieldValue) bool {
		for _, val := range partitionVals {
			switch keyField.DataType {
			case schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64:
				targetVal := storage.NewInt64FieldValue(val.GetInt64Val())
				if targetVal.GT(max) || targetVal.LT(min) {
					return true
				}
			case schemapb.DataType_String, schemapb.DataType_VarChar:
				targetVal := storage.NewVarCharFieldValue(val.String())
				if targetVal.GT(max) || targetVal.LT(min) {
					return true
				}
			}
		}
		return false
	}
	for segID, segStats := range partitionStats.SegmentStats {
		for _, fieldStat := range segStats.FieldStats {
			if keyField.FieldID == fieldStat.FieldID && outRange(fieldStat.Min, fieldStat.Max) {
				filteredSegments[segID] = struct{}{}
			}
		}
	}
}
