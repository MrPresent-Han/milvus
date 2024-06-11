// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package segments

/*
#cgo pkg-config: milvus_segcore

#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"
*/
import "C"

import (
	"context"
	"fmt"
	"math"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"
)

type SliceInfo struct {
	SliceNQs   []int64
	SliceTopKs []int64
}

func ParseSliceInfo(originNQs []int64, originTopKs []int64, nqPerSlice int64) *SliceInfo {
	sInfo := &SliceInfo{
		SliceNQs:   make([]int64, 0),
		SliceTopKs: make([]int64, 0),
	}

	if nqPerSlice == 0 {
		return sInfo
	}

	for i := 0; i < len(originNQs); i++ {
		for j := 0; j < int(originNQs[i]/nqPerSlice); j++ {
			sInfo.SliceNQs = append(sInfo.SliceNQs, nqPerSlice)
			sInfo.SliceTopKs = append(sInfo.SliceTopKs, originTopKs[i])
		}
		if tailSliceSize := originNQs[i] % nqPerSlice; tailSliceSize > 0 {
			sInfo.SliceNQs = append(sInfo.SliceNQs, tailSliceSize)
			sInfo.SliceTopKs = append(sInfo.SliceTopKs, originTopKs[i])
		}
	}

	return sInfo
}

func ReduceSearchResultsAndFillData(ctx context.Context, plan *segments.SearchPlan, searchResults []*segments.SearchResult,
	numSegments int64, sliceNQs []int64, sliceTopKs []int64,
) (segments.SearchResultDataBlobs, error) {
	if plan.GetCSearchPlan() == nil {
		return nil, fmt.Errorf("nil search plan")
	}

	if len(sliceNQs) == 0 {
		return nil, fmt.Errorf("empty slice nqs is not allowed")
	}

	if len(sliceNQs) != len(sliceTopKs) {
		return nil, fmt.Errorf("unaligned sliceNQs(len=%d) and sliceTopKs(len=%d)", len(sliceNQs), len(sliceTopKs))
	}

	cSearchResults := make([]C.CSearchResult, 0)
	for _, res := range searchResults {
		if res == nil {
			return nil, fmt.Errorf("nil searchResult detected when reduceSearchResultsAndFillData")
		}
		cSearchResults = append(cSearchResults, res)
	}
	cSearchResultPtr := &cSearchResults[0]
	cNumSegments := C.int64_t(numSegments)
	cSliceNQSPtr := (*C.int64_t)(&sliceNQs[0])
	cSliceTopKSPtr := (*C.int64_t)(&sliceTopKs[0])
	cNumSlices := C.int64_t(len(sliceNQs))
	var cSearchResultDataBlobs segments.SearchResultDataBlobs
	traceCtx := segments.ParseCTraceContext(ctx)
	status := C.ReduceSearchResultsAndFillData(traceCtx.GetCTraceCtx(), &cSearchResultDataBlobs, plan.GetCSearchPlan(), cSearchResultPtr,
		cNumSegments, cSliceNQSPtr, cSliceTopKSPtr, cNumSlices)
	if err := segments.HandleCStatus(ctx, &status, "ReduceSearchResultsAndFillData failed"); err != nil {
		return nil, err
	}
	return cSearchResultDataBlobs, nil
}

func GetSearchResultDataBlob(ctx context.Context, cSearchResultDataBlobs segments.SearchResultDataBlobs, blobIndex int) ([]byte, error) {
	var blob C.CProto
	status := C.GetSearchResultDataBlob(&blob, cSearchResultDataBlobs, C.int32_t(blobIndex))
	if err := segments.HandleCStatus(ctx, &status, "marshal failed"); err != nil {
		return nil, err
	}
	return segments.GetCProtoBlob(&blob), nil
}

func ReduceSearchResults(ctx context.Context, results []*internalpb.SearchResults, nq int64, topk int64, metricType string) (*internalpb.SearchResults, error) {
	results = lo.Filter(results, func(result *internalpb.SearchResults, _ int) bool {
		return result != nil && result.GetSlicedBlob() != nil
	})

	if len(results) == 1 {
		return results[0], nil
	}

	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "ReduceSearchResults")
	defer sp.End()

	channelsMvcc := make(map[string]uint64)
	for _, r := range results {
		for ch, ts := range r.GetChannelsMvcc() {
			channelsMvcc[ch] = ts
		}
		// shouldn't let new SearchResults.MetricType to be empty, though the req.MetricType is empty
		if metricType == "" {
			metricType = r.MetricType
		}
	}
	log := log.Ctx(ctx)

	searchResultData, err := segments.DecodeSearchResults(ctx, results)
	if err != nil {
		log.Warn("shard leader decode search results errors", zap.Error(err))
		return nil, err
	}
	log.Debug("shard leader get valid search results", zap.Int("numbers", len(searchResultData)))

	for i, sData := range searchResultData {
		log.Debug("reduceSearchResultData",
			zap.Int("result No.", i),
			zap.Int64("nq", sData.NumQueries),
			zap.Int64("topk", sData.TopK))
	}

	reducedResultData, err := ReduceSearchResultData(ctx, searchResultData, nq, topk)
	if err != nil {
		log.Warn("shard leader reduce errors", zap.Error(err))
		return nil, err
	}
	searchResults, err := segments.EncodeSearchResultData(ctx, reducedResultData, nq, topk, metricType)
	if err != nil {
		log.Warn("shard leader encode search result errors", zap.Error(err))
		return nil, err
	}

	requestCosts := lo.FilterMap(results, func(result *internalpb.SearchResults, _ int) (*internalpb.CostAggregation, bool) {
		if paramtable.Get().QueryNodeCfg.EnableWorkerSQCostMetrics.GetAsBool() {
			return result.GetCostAggregation(), true
		}

		if result.GetBase().GetSourceID() == paramtable.GetNodeID() {
			return result.GetCostAggregation(), true
		}

		return nil, false
	})
	searchResults.CostAggregation = segments.MergeRequestCost(requestCosts)
	if searchResults.CostAggregation == nil {
		searchResults.CostAggregation = &internalpb.CostAggregation{}
	}
	relatedDataSize := lo.Reduce(results, func(acc int64, result *internalpb.SearchResults, _ int) int64 {
		return acc + result.GetCostAggregation().GetTotalRelatedDataSize()
	}, 0)
	searchResults.CostAggregation.TotalRelatedDataSize = relatedDataSize
	searchResults.ChannelsMvcc = channelsMvcc
	return searchResults, nil
}

func SelectSearchResultData(dataArray []*schemapb.SearchResultData, resultOffsets [][]int64, offsets []int64, qi int64) int {
	var (
		sel                 = -1
		maxDistance         = -float32(math.MaxFloat32)
		resultDataIdx int64 = -1
	)
	for i, offset := range offsets { // query num, the number of ways to merge
		if offset >= dataArray[i].Topks[qi] {
			continue
		}

		idx := resultOffsets[i][qi] + offset
		distance := dataArray[i].Scores[idx]

		if distance > maxDistance {
			sel = i
			maxDistance = distance
			resultDataIdx = idx
		} else if distance == maxDistance {
			if sel == -1 {
				// A bad case happens where knowhere returns distance == +/-maxFloat32
				// by mistake.
				log.Warn("a bad distance is found, something is wrong here!", zap.Float32("score", distance))
			} else if typeutil.ComparePK(
				typeutil.GetPK(dataArray[i].GetIds(), idx),
				typeutil.GetPK(dataArray[sel].GetIds(), resultDataIdx)) {
				sel = i
				maxDistance = distance
				resultDataIdx = idx
			}
		}
	}
	return sel
}

func ReduceSearchResultData(ctx context.Context, searchResultData []*schemapb.SearchResultData, nq int64, topk int64) (*schemapb.SearchResultData, error) {
	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "ReduceSearchResultData")
	defer sp.End()
	log := log.Ctx(ctx)

	if len(searchResultData) == 0 {
		return &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topk,
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     make([]float32, 0),
			Ids:        &schemapb.IDs{},
			Topks:      make([]int64, 0),
		}, nil
	}
	ret := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: make([]*schemapb.FieldData, len(searchResultData[0].FieldsData)),
		Scores:     make([]float32, 0),
		Ids:        &schemapb.IDs{},
		Topks:      make([]int64, 0),
	}

	resultOffsets := make([][]int64, len(searchResultData))
	for i := 0; i < len(searchResultData); i++ {
		resultOffsets[i] = make([]int64, len(searchResultData[i].Topks))
		for j := int64(1); j < nq; j++ {
			resultOffsets[i][j] = resultOffsets[i][j-1] + searchResultData[i].Topks[j-1]
		}
		ret.AllSearchCount += searchResultData[i].GetAllSearchCount()
	}

	var skipDupCnt int64
	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	for i := int64(0); i < nq; i++ {
		offsets := make([]int64, len(searchResultData))

		idSet := make(map[interface{}]struct{})
		groupByValueSet := make(map[interface{}]struct{})
		var j int64
		for j = 0; j < topk; {
			sel := SelectSearchResultData(searchResultData, resultOffsets, offsets, i)
			if sel == -1 {
				break
			}
			idx := resultOffsets[sel][i] + offsets[sel]

			id := typeutil.GetPK(searchResultData[sel].GetIds(), idx)
			groupByVal := typeutil.GetData(searchResultData[sel].GetGroupByFieldValue(), int(idx))
			score := searchResultData[sel].Scores[idx]

			// remove duplicates
			if _, ok := idSet[id]; !ok {
				groupByValExist := false
				if groupByVal != nil {
					_, groupByValExist = groupByValueSet[groupByVal]
				}
				if !groupByValExist {
					retSize += typeutil.AppendFieldData(ret.FieldsData, searchResultData[sel].FieldsData, idx)
					typeutil.AppendPKs(ret.Ids, id)
					ret.Scores = append(ret.Scores, score)
					if groupByVal != nil {
						groupByValueSet[groupByVal] = struct{}{}
						if err := typeutil.AppendGroupByValue(ret, groupByVal, searchResultData[sel].GetGroupByFieldValue().GetType()); err != nil {
							log.Error("Failed to append groupByValues", zap.Error(err))
							return ret, err
						}
					}
					idSet[id] = struct{}{}
					j++
				}
			} else {
				// skip entity with same id
				skipDupCnt++
			}
			offsets[sel]++
		}

		// if realTopK != -1 && realTopK != j {
		// 	log.Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
		// 	// return nil, errors.New("the length (topk) between all result of query is different")
		// }
		ret.Topks = append(ret.Topks, j)

		// limit search result to avoid oom
		if retSize > maxOutputSize {
			return nil, fmt.Errorf("search results exceed the maxOutputSize Limit %d", maxOutputSize)
		}
	}
	log.Debug("skip duplicated search result", zap.Int64("count", skipDupCnt))
	return ret, nil
}
