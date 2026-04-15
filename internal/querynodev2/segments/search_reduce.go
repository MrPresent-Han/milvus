package segments

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

type SearchReduce interface {
	ReduceSearchResultData(ctx context.Context, searchResultData []*schemapb.SearchResultData, info *reduce.ResultInfo) (*schemapb.SearchResultData, error)
}

type SearchCommonReduce struct{}

func (scr *SearchCommonReduce) ReduceSearchResultData(ctx context.Context, searchResultData []*schemapb.SearchResultData, info *reduce.ResultInfo) (*schemapb.SearchResultData, error) {
	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "ReduceSearchResultData")
	defer sp.End()
	log := log.Ctx(ctx)

	if len(searchResultData) == 0 {
		return &schemapb.SearchResultData{
			NumQueries: info.GetNq(),
			TopK:       info.GetTopK(),
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     make([]float32, 0),
			Ids:        &schemapb.IDs{},
			Topks:      make([]int64, 0),
		}, nil
	}
	nq := info.GetNq()
	topk := info.GetTopK()
	ret := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: make([]*schemapb.FieldData, len(searchResultData[0].FieldsData)),
		Scores:     make([]float32, 0, nq*topk),
		Ids:        &schemapb.IDs{},
		Topks:      make([]int64, 0, nq),
	}

	// Determine element-level flag: any result having ElementIndices means element-level search.
	hasElementIndices := false
	for _, data := range searchResultData {
		if data.ElementIndices != nil {
			hasElementIndices = true
			break
		}
	}
	if hasElementIndices {
		for i, data := range searchResultData {
			// If any result has ElementIndices, all results must have ElementIndices or no doc is hit in the segment
			if data.ElementIndices == nil {
				ids := data.GetIds()
				// When a segment returns 0 hits, C++ reduce creates an empty LongArray
				// which proto3 serializes as absent (nil). Back-fill for uniformity.
				if ids == nil || typeutil.GetSizeOfIDs(ids) == 0 {
					data.ElementIndices = &schemapb.LongArray{}
				} else {
					return nil, fmt.Errorf("inconsistent element-level flag in search results: result has data but missing ElementIndices at index %d", i)
				}
			}
		}
		ret.ElementIndices = &schemapb.LongArray{
			Data: make([]int64, 0),
		}
	}

	resultOffsets := make([][]int64, len(searchResultData))
	totalOffsetElements := 0
	for i := range searchResultData {
		totalOffsetElements += len(searchResultData[i].Topks)
	}
	offsetBacking := make([]int64, totalOffsetElements)
	for i := 0; i < len(searchResultData); i++ {
		n := len(searchResultData[i].Topks)
		resultOffsets[i] = offsetBacking[:n:n]
		offsetBacking = offsetBacking[n:]
		for j := int64(1); j < nq; j++ {
			resultOffsets[i][j] = resultOffsets[i][j-1] + searchResultData[i].Topks[j-1]
		}
		ret.AllSearchCount += searchResultData[i].GetAllSearchCount()
	}

	idxComputers := make([]*typeutil.FieldDataIdxComputer, len(searchResultData))
	for i, srd := range searchResultData {
		idxComputers[i] = typeutil.NewFieldDataIdxComputer(srd.FieldsData)
	}

	numResults := len(searchResultData)
	var skipDupCnt int64
	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	for i := int64(0); i < nq; i++ {
		offsets := make([]int64, numResults)
		idSet := make(map[interface{}]struct{})
		var j int64
		for j = 0; j < topk; {
			sel := SelectSearchResultData(searchResultData, resultOffsets, offsets, i)
			if sel == -1 {
				break
			}
			idx := resultOffsets[sel][i] + offsets[sel]

			id := typeutil.GetPK(searchResultData[sel].GetIds(), idx)
			score := searchResultData[sel].Scores[idx]

			// remove duplicates
			if _, ok := idSet[id]; !ok {
				fieldsData := searchResultData[sel].FieldsData
				fieldIdxs := idxComputers[sel].Compute(idx)
				retSize += typeutil.AppendFieldData(ret.FieldsData, fieldsData, idx, fieldIdxs...)
				typeutil.AppendPKs(ret.Ids, id)
				ret.Scores = append(ret.Scores, score)
				if searchResultData[sel].ElementIndices != nil && ret.ElementIndices != nil {
					ret.ElementIndices.Data = append(ret.ElementIndices.Data, searchResultData[sel].ElementIndices.Data[idx])
				}
				idSet[id] = struct{}{}
				j++
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

type SearchGroupByReduce struct{}

func (sbr *SearchGroupByReduce) ReduceSearchResultData(ctx context.Context, searchResultData []*schemapb.SearchResultData, info *reduce.ResultInfo) (*schemapb.SearchResultData, error) {
	ctx, sp := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "ReduceSearchResultData")
	defer sp.End()
	log := log.Ctx(ctx)

	if len(searchResultData) == 0 {
		log.Debug("Shortcut return SearchGroupByReduce, directly return empty result", zap.Any("result info", info))
		return &schemapb.SearchResultData{
			NumQueries: info.GetNq(),
			TopK:       info.GetTopK(),
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     make([]float32, 0),
			Ids:        &schemapb.IDs{},
			Topks:      make([]int64, 0),
		}, nil
	}
	nq := info.GetNq()
	topk := info.GetTopK()
	ret := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: make([]*schemapb.FieldData, len(searchResultData[0].FieldsData)),
		Scores:     make([]float32, 0, nq*topk),
		Ids:        &schemapb.IDs{},
		Topks:      make([]int64, 0, nq),
	}

	// Determine element-level flag: any result having ElementIndices means element-level search.
	hasElementIndices := false
	for _, data := range searchResultData {
		if data.ElementIndices != nil {
			hasElementIndices = true
			break
		}
	}
	if hasElementIndices {
		// If any result has ElementIndices, all results must have ElementIndices or no doc is hit in the segment
		for i, data := range searchResultData {
			if data.ElementIndices == nil {
				ids := data.GetIds()
				// When a segment returns 0 hits, C++ reduce creates an empty LongArray
				// which proto3 serializes as absent (nil). Back-fill for uniformity.
				if ids == nil || typeutil.GetSizeOfIDs(ids) == 0 {
					data.ElementIndices = &schemapb.LongArray{}
				} else {
					return nil, fmt.Errorf("inconsistent element-level flag in search results: result has data but missing ElementIndices at index %d", i)
				}
			}
		}
		ret.ElementIndices = &schemapb.LongArray{
			Data: make([]int64, 0),
		}
	}

	resultOffsets := make([][]int64, len(searchResultData))
	totalOffsetElements := 0
	for i := range searchResultData {
		totalOffsetElements += len(searchResultData[i].Topks)
	}
	offsetBacking := make([]int64, totalOffsetElements)
	for i := range searchResultData {
		n := len(searchResultData[i].Topks)
		resultOffsets[i] = offsetBacking[:n:n]
		offsetBacking = offsetBacking[n:]
		for j := int64(1); j < nq; j++ {
			resultOffsets[i][j] = resultOffsets[i][j-1] + searchResultData[i].Topks[j-1]
		}
		ret.AllSearchCount += searchResultData[i].GetAllSearchCount()
	}

	keyExtractors := buildKeyExtractors(searchResultData, info)
	idxComputers := make([]*typeutil.FieldDataIdxComputer, len(searchResultData))
	for i, srd := range searchResultData {
		idxComputers[i] = typeutil.NewFieldDataIdxComputer(srd.FieldsData)
	}

	var filteredCount int64
	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	groupSize := info.GetGroupSize()
	if groupSize <= 0 {
		groupSize = 1
	}
	groupBound := info.GetTopK() * groupSize
	acceptedRows := make([]reduce.RowRef, 0, nq*groupBound)

	for i := int64(0); i < info.GetNq(); i++ {
		offsets := make([]int64, len(searchResultData))

		idSet := make(map[interface{}]struct{})
		// groupBuckets stores per-composite-key state keyed by uint64 hash.
		// Collisions resolve via linear scan over the per-bucket slice, just
		// like internal/agg's reducer — cheap given normal collision rates.
		groupBuckets := make(map[uint64][]*reduceGroupEntry)
		totalGroups := int64(0)

		var j int64
		for j = 0; j < groupBound; {
			sel := SelectSearchResultData(searchResultData, resultOffsets, offsets, i)
			if sel == -1 {
				break
			}
			idx := resultOffsets[sel][i] + offsets[sel]

			id := typeutil.GetPK(searchResultData[sel].GetIds(), idx)
			hash, values := keyExtractors[sel](int(idx))
			score := searchResultData[sel].Scores[idx]
			if _, ok := idSet[id]; !ok {
				entry := findReduceGroupEntry(groupBuckets[hash], values)
				isNewGroup := entry == nil
				if isNewGroup && totalGroups >= info.GetTopK() {
					// exceed the limit for group count, filter this entity
					filteredCount++
				} else if !isNewGroup && entry.count >= groupSize {
					// exceed the limit for each group, filter this entity
					filteredCount++
				} else {
					if isNewGroup {
						entry = &reduceGroupEntry{values: values}
						groupBuckets[hash] = append(groupBuckets[hash], entry)
						totalGroups++
					}
					fieldsData := searchResultData[sel].FieldsData
					fieldIdxs := idxComputers[sel].Compute(idx)
					retSize += typeutil.AppendFieldData(ret.FieldsData, fieldsData, idx, fieldIdxs...)
					typeutil.AppendPKs(ret.Ids, id)
					ret.Scores = append(ret.Scores, score)
					if searchResultData[sel].ElementIndices != nil && ret.ElementIndices != nil {
						ret.ElementIndices.Data = append(ret.ElementIndices.Data, searchResultData[sel].ElementIndices.Data[idx])
					}
					entry.count++
					idSet[id] = struct{}{}
					acceptedRows = append(acceptedRows, reduce.RowRef{ResultIdx: sel, RowIdx: idx})
					j++
				}
			} else {
				// skip entity with same pk
				filteredCount++
			}
			offsets[sel]++
		}
		ret.Topks = append(ret.Topks, j)

		// limit search result to avoid oom
		if retSize > maxOutputSize {
			return nil, fmt.Errorf("search results exceed the maxOutputSize Limit %d", maxOutputSize)
		}
	}
	if err := writeGroupByOutput(ret, acceptedRows, searchResultData, info); err != nil {
		return ret, merr.WrapErrServiceInternal("failed to construct group by output", err.Error())
	}
	if float64(filteredCount) >= 0.3*float64(groupBound) {
		log.Warn("GroupBy reduce filtered too many results, "+
			"this may influence the final result seriously",
			zap.Int64("filteredCount", filteredCount),
			zap.Int64("groupBound", groupBound))
	}
	log.Debug("skip duplicated search result", zap.Int64("count", filteredCount))
	return ret, nil
}

func InitSearchReducer(info *reduce.ResultInfo) SearchReduce {
	if info.GetGroupByFieldId() > 0 || info.HasMultiGroupBy() {
		return &SearchGroupByReduce{}
	}
	return &SearchCommonReduce{}
}

// reduceGroupEntry tracks one composite-key bucket during SearchGroupByReduce.
// Held inside a map[uint64][]*reduceGroupEntry keyed by hash, so collisions
// resolve via linear scan with reduce.EqualGroupValues.
type reduceGroupEntry struct {
	values []any
	count  int64
}

func findReduceGroupEntry(bucket []*reduceGroupEntry, values []any) *reduceGroupEntry {
	for _, e := range bucket {
		if reduce.EqualGroupValues(e.values, values) {
			return e
		}
	}
	return nil
}

// keyExtractor returns (hash, normalized values) for the row at idx. The hash
// map drives bucket lookup; the values slice is retained for hash-collision
// disambiguation via reduce.EqualGroupValues. Values are normalized through
// reduce.NormalizeScalar so types align with the proxy-side reducer.
type keyExtractor func(idx int) (uint64, []any)

func buildKeyExtractors(searchResultData []*schemapb.SearchResultData, info *reduce.ResultInfo) []keyExtractor {
	if info.HasMultiGroupBy() {
		return buildMultiFieldExtractors(searchResultData, info.GetMultiGroupByFieldIds())
	}
	return buildSingleFieldExtractors(searchResultData)
}

func buildSingleFieldExtractors(searchResultData []*schemapb.SearchResultData) []keyExtractor {
	// Single-field group-by is the N=1 specialization of the multi-field
	// extractor. Wrap the GroupByFieldValue iterator as a 1-element iters
	// slice and reuse the shared factory.
	extractors := make([]keyExtractor, len(searchResultData))
	for i := range searchResultData {
		iter := typeutil.GetDataIterator(searchResultData[i].GetGroupByFieldValue())
		extractors[i] = reduce.MakeCompositeKeyExtractor([]func(int) any{iter})
	}
	return extractors
}

func buildMultiFieldExtractors(searchResultData []*schemapb.SearchResultData, groupByFieldIDs []int64) []keyExtractor {
	extractors := make([]keyExtractor, len(searchResultData))
	for i := range searchResultData {
		iters := make([]func(int) any, len(groupByFieldIDs))
		for j, fieldID := range groupByFieldIDs {
			// Group-by columns ride on proto field 17, separate from FieldsData.
			fieldData := reduce.FindFieldDataByID(searchResultData[i].GetGroupByFieldValues(), fieldID)
			if fieldData != nil {
				iters[j] = typeutil.GetDataIterator(fieldData)
			}
		}
		extractors[i] = reduce.MakeCompositeKeyExtractor(iters)
	}
	return extractors
}

func writeGroupByOutput(ret *schemapb.SearchResultData, acceptedRows []reduce.RowRef, searchResultData []*schemapb.SearchResultData, info *reduce.ResultInfo) error {
	if info.HasMultiGroupBy() {
		return reduce.WriteGroupByFieldValues(ret, acceptedRows, searchResultData, info.GetMultiGroupByFieldIds())
	}
	if len(acceptedRows) == 0 {
		if len(searchResultData) == 0 || searchResultData[0].GetGroupByFieldValue() == nil {
			ret.GroupByFieldValue = nil
			return nil
		}
		gpFieldBuilder, err := typeutil.NewFieldDataBuilder(searchResultData[0].GetGroupByFieldValue().GetType(), true, 0)
		if err != nil {
			return err
		}
		ret.GroupByFieldValue = gpFieldBuilder.Build()
		return nil
	}
	baseField := searchResultData[acceptedRows[0].ResultIdx].GetGroupByFieldValue()
	if baseField == nil {
		ret.GroupByFieldValue = nil
		return nil
	}
	gpFieldBuilder, err := typeutil.NewFieldDataBuilder(baseField.GetType(), true, len(acceptedRows))
	if err != nil {
		return err
	}
	iterators := make([]func(int) any, len(searchResultData))
	for i := range searchResultData {
		iterators[i] = typeutil.GetDataIterator(searchResultData[i].GetGroupByFieldValue())
	}
	for _, rowRef := range acceptedRows {
		gpFieldBuilder.Add(iterators[rowRef.ResultIdx](int(rowRef.RowIdx)))
	}
	ret.GroupByFieldValue = gpFieldBuilder.Build()
	return nil
}
