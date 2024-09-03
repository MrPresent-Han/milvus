package proxy

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus/internal/util/reduce"
	"math"
	"sort"

	"github.com/cockroachdb/errors"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metric"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

func reduceSearchResult(ctx context.Context, subSearchResultData []*schemapb.SearchResultData, reduceInfo *reduce.ResultInfo) (*milvuspb.SearchResults, error) {
	if reduceInfo.GetGroupByFieldId() > 0 {
		if reduceInfo.GetIsAdvance() {
			//for hybrid search groupby, we cannot reduce result for results from one single search path,
			//because the final score has not been accumulated, also, offset cannot be applied
			return reduceAdvanceGroupBY(ctx,
				subSearchResultData, reduceInfo.GetNq(), reduceInfo.GetTopK(), reduceInfo.GetPkType(), reduceInfo.GetMetricType())
		} else {
			return reduceSearchResultDataWithGroupBy(ctx,
				subSearchResultData,
				reduceInfo.GetNq(),
				reduceInfo.GetTopK(),
				reduceInfo.GetMetricType(),
				reduceInfo.GetPkType(),
				reduceInfo.GetOffset(),
				reduceInfo.GetGroupSize())
		}
	}
	return reduceSearchResultDataNoGroupBy(ctx,
		subSearchResultData,
		reduceInfo.GetNq(),
		reduceInfo.GetTopK(),
		reduceInfo.GetMetricType(),
		reduceInfo.GetPkType(),
		reduceInfo.GetOffset())
}

func checkResultDatas(ctx context.Context, subSearchResultData []*schemapb.SearchResultData,
	nq int64, topK int64) (int64, int, error) {
	var allSearchCount int64
	var hitNum int
	for i, sData := range subSearchResultData {
		pkLength := typeutil.GetSizeOfIDs(sData.GetIds())
		log.Ctx(ctx).Debug("subSearchResultData",
			zap.Int("result No.", i),
			zap.Int64("nq", sData.NumQueries),
			zap.Int64("topk", sData.TopK),
			zap.Int("length of pks", pkLength),
			zap.Int("length of FieldsData", len(sData.FieldsData)))
		allSearchCount += sData.GetAllSearchCount()
		hitNum += pkLength
		if err := checkSearchResultData(sData, nq, topK, pkLength); err != nil {
			log.Ctx(ctx).Warn("invalid search results", zap.Error(err))
			return allSearchCount, hitNum, err
		}
	}
	return allSearchCount, hitNum, nil
}

func reduceAdvanceGroupBY(ctx context.Context, subSearchResultData []*schemapb.SearchResultData,
	nq int64, topK int64, pkType MilvusPKType, metricType string) (*milvuspb.SearchResults, error) {
	log.Ctx(ctx).Debug("reduceAdvanceGroupBY", zap.Int("len(subSearchResultData)", len(subSearchResultData)), zap.Int64("nq", nq))
	// for advance group by, offset is not applied, so just return when there's only one channel
	if len(subSearchResultData) == 1 {
		return &milvuspb.SearchResults{
			Status:  merr.Success(),
			Results: subSearchResultData[0],
		}, nil
	}

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topK,
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}

	var limit int64
	if allSearchCount, hitNum, err := checkResultDatas(ctx, subSearchResultData, nq, topK); err != nil {
		log.Ctx(ctx).Warn("invalid search results", zap.Error(err))
		return ret, err
	} else {
		ret.GetResults().AllSearchCount = allSearchCount
		limit = int64(hitNum)
		ret.GetResults().FieldsData = typeutil.PrepareResultFieldData(subSearchResultData[0].GetFieldsData(), limit)
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0, limit),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0, limit),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	var (
		subSearchNum = len(subSearchResultData)
		// for results of each subSearchResultData, storing the start offset of each query of nq queries
		subSearchNqOffset = make([][]int64, subSearchNum)
	)
	for i := 0; i < subSearchNum; i++ {
		subSearchNqOffset[i] = make([]int64, subSearchResultData[i].GetNumQueries())
		for j := int64(1); j < nq; j++ {
			subSearchNqOffset[i][j] = subSearchNqOffset[i][j-1] + subSearchResultData[i].Topks[j-1]
		}
	}

	var (
		skipDupCnt int64
		realTopK   int64 = -1
	)

	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()

	// reducing nq * topk results
	for i := int64(0); i < nq; i++ {
		var (
			// cursor of current data of each subSearch for merging the j-th data of TopK.
			// sum(cursors) == j
			cursors = make([]int64, subSearchNum)

			j     int64
			idSet = make(map[interface{}]struct{}, limit)
		)

		// keep limit results
		for j = 0; j < limit; {
			// From all the sub-query result sets of the i-th query vector,
			//   find the sub-query result set index of the score j-th data,
			//   and the index of the data in schemapb.SearchResultData
			subSearchIdx, resultDataIdx := selectHighestScoreIndex(subSearchResultData, subSearchNqOffset, cursors, i)
			if subSearchIdx == -1 {
				break
			}
			id := typeutil.GetPK(subSearchResultData[subSearchIdx].GetIds(), resultDataIdx)
			score := subSearchResultData[subSearchIdx].Scores[resultDataIdx]
			subSearchRes := subSearchResultData[subSearchIdx]
			groupByVal := typeutil.GetData(subSearchRes.GetGroupByFieldValue(), int(resultDataIdx))
			// remove duplicatessds
			if _, ok := idSet[id]; !ok {
				retSize += typeutil.AppendFieldData(ret.Results.FieldsData, subSearchResultData[subSearchIdx].FieldsData, resultDataIdx)
				typeutil.AppendPKs(ret.Results.Ids, id)
				ret.Results.Scores = append(ret.Results.Scores, score)
				idSet[id] = struct{}{}
				j++
				if err := typeutil.AppendGroupByValue(ret.Results, groupByVal, subSearchRes.GetGroupByFieldValue().GetType()); err != nil {
					log.Ctx(ctx).Error("failed to append groupByValues", zap.Error(err))
					return ret, err
				}
			} else {
				// skip entity with same id
				skipDupCnt++
			}
			cursors[subSearchIdx]++
		}
		if realTopK != -1 && realTopK != j {
			log.Ctx(ctx).Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
			// return nil, errors.New("the length (topk) between all result of query is different")
		}
		realTopK = j
		ret.Results.Topks = append(ret.Results.Topks, realTopK)

		// limit search result to avoid oom
		if retSize > maxOutputSize {
			return nil, fmt.Errorf("search results exceed the maxOutputSize Limit %d", maxOutputSize)
		}
	}
	log.Ctx(ctx).Debug("skip duplicated search result", zap.Int64("count", skipDupCnt))

	if skipDupCnt > 0 {
		log.Info("skip duplicated search result", zap.Int64("count", skipDupCnt))
	}

	ret.Results.TopK = realTopK // realTopK is the topK of the nq-th query
	if !metric.PositivelyRelated(metricType) {
		for k := range ret.Results.Scores {
			ret.Results.Scores[k] *= -1
		}
	}
	return ret, nil
}

type MilvusPKType interface{}

type groupReduceInfo struct {
	subSearchIdx int
	resultIdx    int64
	score        float32
	id           MilvusPKType
}

func reduceSearchResultDataWithGroupBy(ctx context.Context, subSearchResultData []*schemapb.SearchResultData,
	nq int64, topk int64, metricType string,
	pkType schemapb.DataType,
	offset int64,
	groupSize int64,
) (*milvuspb.SearchResults, error) {
	tr := timerecord.NewTimeRecorder("reduceSearchResultData")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	limit := topk - offset
	log.Ctx(ctx).Debug("reduceSearchResultData",
		zap.Int("len(subSearchResultData)", len(subSearchResultData)),
		zap.Int64("nq", nq),
		zap.Int64("offset", offset),
		zap.Int64("limit", limit),
		zap.String("metricType", metricType))

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topk,
			FieldsData: typeutil.PrepareResultFieldData(subSearchResultData[0].GetFieldsData(), limit),
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0, limit),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0, limit),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	if allSearchCount, _, err := checkResultDatas(ctx, subSearchResultData, nq, topk); err != nil {
		log.Ctx(ctx).Warn("invalid search results", zap.Error(err))
		return ret, err
	} else {
		ret.GetResults().AllSearchCount = allSearchCount
	}

	var (
		subSearchNum = len(subSearchResultData)
		// for results of each subSearchResultData, storing the start offset of each query of nq queries
		subSearchNqOffset       = make([][]int64, subSearchNum)
		totalResCount     int64 = 0
	)
	for i := 0; i < subSearchNum; i++ {
		subSearchNqOffset[i] = make([]int64, subSearchResultData[i].GetNumQueries())
		for j := int64(1); j < nq; j++ {
			subSearchNqOffset[i][j] = subSearchNqOffset[i][j-1] + subSearchResultData[i].Topks[j-1]
		}
		totalResCount += subSearchNqOffset[i][nq-1]
	}

	var (
		skipDupCnt int64
		realTopK   int64 = -1
	)

	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()
	groupBound := groupSize * limit

	// reducing nq * topk results
	for i := int64(0); i < nq; i++ {
		var (
			// cursor of current data of each subSearch for merging the j-th data of TopK.
			// sum(cursors) == j
			cursors = make([]int64, subSearchNum)

			j              int64
			pkSet          = make(map[interface{}]struct{})
			groupByValMap  = make(map[interface{}][]*groupReduceInfo)
			skipOffsetMap  = make(map[interface{}]bool)
			groupByValList = make([]interface{}, limit)
			groupByValIdx  = 0
		)

		for j = 0; j < groupBound; {
			subSearchIdx, resultDataIdx := selectHighestScoreIndex(subSearchResultData, subSearchNqOffset, cursors, i)
			if subSearchIdx == -1 {
				break
			}
			subSearchRes := subSearchResultData[subSearchIdx]

			id := typeutil.GetPK(subSearchRes.GetIds(), resultDataIdx)
			score := subSearchRes.GetScores()[resultDataIdx]
			groupByVal := typeutil.GetData(subSearchRes.GetGroupByFieldValue(), int(resultDataIdx))
			if groupByVal == nil {
				return nil, errors.New("get nil groupByVal from subSearchRes, wrong states, as milvus doesn't support nil value," +
					"there must be sth wrong on queryNode side")
			}
			// remove duplicates
			if _, ok := pkSet[id]; !ok {
				if int64(len(skipOffsetMap)) < offset || skipOffsetMap[groupByVal] {
					skipOffsetMap[groupByVal] = true
					// the first offset's group will be ignored
					skipDupCnt++
				} else if len(groupByValMap[groupByVal]) == 0 && int64(len(groupByValMap)) >= limit {
					// skip when groupbyMap has been full and found new groupByVal
					skipDupCnt++
				} else if int64(len(groupByValMap[groupByVal])) >= groupSize {
					// skip when target group has been full
					skipDupCnt++
				} else {
					if len(groupByValMap[groupByVal]) == 0 {
						groupByValList[groupByValIdx] = groupByVal
						groupByValIdx++
					}
					groupByValMap[groupByVal] = append(groupByValMap[groupByVal], &groupReduceInfo{
						subSearchIdx: subSearchIdx,
						resultIdx:    resultDataIdx, id: id, score: score,
					})
					pkSet[id] = struct{}{}
					j++
				}
			} else {
				skipDupCnt++
			}

			cursors[subSearchIdx]++
		}

		// assemble all eligible values in group
		// values in groupByValList is sorted by the highest score in each group
		for _, groupVal := range groupByValList {
			if groupVal != nil {
				groupEntities := groupByValMap[groupVal]
				for _, groupEntity := range groupEntities {
					subResData := subSearchResultData[groupEntity.subSearchIdx]
					retSize += typeutil.AppendFieldData(ret.Results.FieldsData, subResData.FieldsData, groupEntity.resultIdx)
					typeutil.AppendPKs(ret.Results.Ids, groupEntity.id)
					ret.Results.Scores = append(ret.Results.Scores, groupEntity.score)
					if err := typeutil.AppendGroupByValue(ret.Results, groupVal, subResData.GetGroupByFieldValue().GetType()); err != nil {
						log.Ctx(ctx).Error("failed to append groupByValues", zap.Error(err))
						return ret, err
					}
				}
			}
		}

		if realTopK != -1 && realTopK != j {
			log.Ctx(ctx).Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
		}
		realTopK = j
		ret.Results.Topks = append(ret.Results.Topks, realTopK)

		// limit search result to avoid oom
		if retSize > maxOutputSize {
			return nil, fmt.Errorf("search results exceed the maxOutputSize Limit %d", maxOutputSize)
		}
	}
	log.Ctx(ctx).Debug("skip duplicated search result when doing group by", zap.Int64("count", skipDupCnt))

	if float64(skipDupCnt) >= float64(totalResCount)*0.3 {
		log.Warn("GroupBy reduce skipped too many results, "+
			"this may influence the final result seriously",
			zap.Int64("skipDupCnt", skipDupCnt),
			zap.Int64("groupBound", groupBound))
	}

	ret.Results.TopK = realTopK // realTopK is the topK of the nq-th query
	if !metric.PositivelyRelated(metricType) {
		for k := range ret.Results.Scores {
			ret.Results.Scores[k] *= -1
		}
	}
	return ret, nil
}

func reduceSearchResultDataNoGroupBy(ctx context.Context, subSearchResultData []*schemapb.SearchResultData, nq int64, topk int64, metricType string, pkType schemapb.DataType, offset int64) (*milvuspb.SearchResults, error) {
	tr := timerecord.NewTimeRecorder("reduceSearchResultData")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	limit := topk - offset
	log.Ctx(ctx).Debug("reduceSearchResultData",
		zap.Int("len(subSearchResultData)", len(subSearchResultData)),
		zap.Int64("nq", nq),
		zap.Int64("offset", offset),
		zap.Int64("limit", limit),
		zap.String("metricType", metricType))

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       topk,
			FieldsData: typeutil.PrepareResultFieldData(subSearchResultData[0].GetFieldsData(), limit),
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0, limit),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0, limit),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	if allSearchCount, _, err := checkResultDatas(ctx, subSearchResultData, nq, topk); err != nil {
		log.Ctx(ctx).Warn("invalid search results", zap.Error(err))
		return ret, err
	} else {
		ret.GetResults().AllSearchCount = allSearchCount
	}

	var (
		subSearchNum = len(subSearchResultData)
		// for results of each subSearchResultData, storing the start offset of each query of nq queries
		subSearchNqOffset = make([][]int64, subSearchNum)
	)
	for i := 0; i < subSearchNum; i++ {
		subSearchNqOffset[i] = make([]int64, subSearchResultData[i].GetNumQueries())
		for j := int64(1); j < nq; j++ {
			subSearchNqOffset[i][j] = subSearchNqOffset[i][j-1] + subSearchResultData[i].Topks[j-1]
		}
	}

	var (
		skipDupCnt int64
		realTopK   int64 = -1
	)

	var retSize int64
	maxOutputSize := paramtable.Get().QuotaConfig.MaxOutputSize.GetAsInt64()

	// reducing nq * topk results
	for i := int64(0); i < nq; i++ {
		var (
			// cursor of current data of each subSearch for merging the j-th data of TopK.
			// sum(cursors) == j
			cursors = make([]int64, subSearchNum)

			j     int64
			idSet = make(map[interface{}]struct{}, limit)
		)

		// skip offset results
		for k := int64(0); k < offset; k++ {
			subSearchIdx, _ := selectHighestScoreIndex(subSearchResultData, subSearchNqOffset, cursors, i)
			if subSearchIdx == -1 {
				break
			}

			cursors[subSearchIdx]++
		}

		// keep limit results
		for j = 0; j < limit; {
			// From all the sub-query result sets of the i-th query vector,
			//   find the sub-query result set index of the score j-th data,
			//   and the index of the data in schemapb.SearchResultData
			subSearchIdx, resultDataIdx := selectHighestScoreIndex(subSearchResultData, subSearchNqOffset, cursors, i)
			if subSearchIdx == -1 {
				break
			}
			id := typeutil.GetPK(subSearchResultData[subSearchIdx].GetIds(), resultDataIdx)
			score := subSearchResultData[subSearchIdx].Scores[resultDataIdx]

			// remove duplicatessds
			if _, ok := idSet[id]; !ok {
				retSize += typeutil.AppendFieldData(ret.Results.FieldsData, subSearchResultData[subSearchIdx].FieldsData, resultDataIdx)
				typeutil.AppendPKs(ret.Results.Ids, id)
				ret.Results.Scores = append(ret.Results.Scores, score)
				idSet[id] = struct{}{}
				j++
			} else {
				// skip entity with same id
				skipDupCnt++
			}
			cursors[subSearchIdx]++
		}
		if realTopK != -1 && realTopK != j {
			log.Ctx(ctx).Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
			// return nil, errors.New("the length (topk) between all result of query is different")
		}
		realTopK = j
		ret.Results.Topks = append(ret.Results.Topks, realTopK)

		// limit search result to avoid oom
		if retSize > maxOutputSize {
			return nil, fmt.Errorf("search results exceed the maxOutputSize Limit %d", maxOutputSize)
		}
	}
	log.Ctx(ctx).Debug("skip duplicated search result", zap.Int64("count", skipDupCnt))

	if skipDupCnt > 0 {
		log.Info("skip duplicated search result", zap.Int64("count", skipDupCnt))
	}

	ret.Results.TopK = realTopK // realTopK is the topK of the nq-th query
	if !metric.PositivelyRelated(metricType) {
		for k := range ret.Results.Scores {
			ret.Results.Scores[k] *= -1
		}
	}
	return ret, nil
}

func rankSearchResultData(ctx context.Context,
	nq int64,
	params *rankParams,
	pkType schemapb.DataType,
	searchResults []*milvuspb.SearchResults,
	groupByFieldID int64,
	groupSize int64,
	groupScorer func(group *Group) error,
) (*milvuspb.SearchResults, error) {
	if groupByFieldID > 0 {
		return rankSearchResultDataByGroup(ctx, nq, params, pkType, searchResults, groupScorer, groupSize)
	}
	return rankSearchResultDataByPk(ctx, nq, params, pkType, searchResults)
}

func GetGroupScorer(scorerType string) (func(group *Group) error, error) {
	switch scorerType {
	case MaxScorer:
		return func(group *Group) error {
			group.finalScore = group.maxScore
			return nil
		}, nil
	case SumScorer:
		return func(group *Group) error {
			group.finalScore = group.sumScore
			return nil
		}, nil
	case AvgScorer:
		return func(group *Group) error {
			if len(group.idList) == 0 {
				return merr.WrapErrParameterInvalid(1, len(group.idList),
					"input group for score must have at least one id, must be sth wrong within code")
			}
			group.finalScore = group.sumScore / float32(len(group.idList))
			return nil
		}, nil
	default:
		return nil, merr.WrapErrParameterInvalidMsg("input group scorer type: %s is not supported!", scorerType)
	}
}

type Group struct {
	idList     []interface{}
	scoreList  []float32
	groupVal   interface{}
	maxScore   float32
	minScore   float32
	sumScore   float32
	finalScore float32
}

func rankSearchResultDataByGroup(ctx context.Context,
	nq int64,
	params *rankParams,
	pkType schemapb.DataType,
	searchResults []*milvuspb.SearchResults,
	groupScorer func(group *Group) error,
	groupSize int64,
) (*milvuspb.SearchResults, error) {
	tr := timerecord.NewTimeRecorder("rankSearchResultDataByGroup")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()
	offset := params.offset
	limit := params.limit
	// in the context of groupby, the meaning for offset refers to the offset of group
	topk := limit + offset
	roundDecimal := params.roundDecimal
	log.Ctx(ctx).Debug("rankSearchResultDataByGroup",
		zap.Int("len(searchResults)", len(searchResults)),
		zap.Int64("nq", nq),
		zap.Int64("offset", offset),
		zap.Int64("limit", limit))

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       limit,
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}
	if len(searchResults) == 0 {
		return ret, nil
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	// []map[id]score
	type accumulateIDGroupVal struct {
		accumulatedScore float32
		groupVal         interface{}
	}

	accumulatedScores := make([]map[interface{}]*accumulateIDGroupVal, nq)
	for i := int64(0); i < nq; i++ {
		accumulatedScores[i] = make(map[interface{}]*accumulateIDGroupVal)
	}

	groupByDataType := searchResults[0].GetResults().GetGroupByFieldValue().GetType()

	for _, result := range searchResults {
		scores := result.GetResults().GetScores()
		start := int64(0)
		for i := int64(0); i < nq; i++ {
			realTopk := result.GetResults().Topks[i]
			for j := start; j < start+realTopk; j++ {
				id := typeutil.GetPK(result.GetResults().GetIds(), j)
				//hc---risky here to convert int64 into int
				groupByVal := typeutil.GetData(result.GetResults().GetGroupByFieldValue(), int(j))
				if accumulatedScores[i][id] != nil {
					accumulatedScores[i][id].accumulatedScore += scores[j]
				} else {
					accumulatedScores[i][id] = &accumulateIDGroupVal{accumulatedScore: scores[j], groupVal: groupByVal}
				}
			}
			start += realTopk
		}
	}

	for i := int64(0); i < nq; i++ {
		idSet := accumulatedScores[i]
		keys := make([]interface{}, 0)
		for key := range idSet {
			keys = append(keys, key)
		}

		compareKeys := func(keyI, keyJ interface{}) bool {
			switch keyI.(type) {
			case int64:
				return keyI.(int64) < keyJ.(int64)
			case string:
				return keyI.(string) < keyJ.(string)
			}
			return false
		}

		// sort id by score
		big := func(i, j int) bool {
			scoreItemI := idSet[keys[i]]
			scoreItemJ := idSet[keys[j]]
			if scoreItemI.accumulatedScore == scoreItemJ.accumulatedScore {
				return compareKeys(keys[i], keys[j])
			}
			return scoreItemI.accumulatedScore > scoreItemJ.accumulatedScore
		}
		sort.Slice(keys, big)

		// separate keys into buckets according to groupVal
		buckets := make(map[interface{}]*Group)
		for _, key := range keys {
			scoreItem := idSet[key]
			groupVal := scoreItem.groupVal
			if buckets[groupVal] == nil {
				buckets[groupVal] = &Group{
					//hc---maybe preallocate here?
					idList:    make([]interface{}, 0),
					scoreList: make([]float32, 0),
					groupVal:  groupVal,
				}
			}
			if int64(len(buckets[groupVal].idList)) >= groupSize {
				//only consider group size results in each group
				continue
			}
			buckets[groupVal].idList = append(buckets[groupVal].idList, key)
			buckets[groupVal].scoreList = append(buckets[groupVal].scoreList, scoreItem.accumulatedScore)
		}
		if int64(len(buckets)) <= offset {
			ret.Results.Topks = append(ret.Results.Topks, 0)
			continue
		}

		groupList := make([]*Group, len(buckets))
		for _, group := range buckets {
			groupScorer(group)
			groupList = append(groupList, group)
		}
		sort.Slice(groupList, func(i, j int) bool {
			if groupList[i].finalScore == groupList[j].finalScore {
				if len(groupList[i].idList) == len(groupList[j].idList) {
					//if final score and size of group are both equal
					//choose the group with smaller first key
					//here, it's guaranteed all group having at least one id in the idList
					return compareKeys(groupList[i].idList[0], groupList[j].idList[0])
				}
				// choose the larger group when scores are equal
				return len(groupList[i].idList) > len(groupList[j].idList)
			}
			return groupList[i].finalScore > groupList[j].finalScore
		})

		if int64(len(groupList)) > topk {
			groupList = groupList[:topk]
		}
		ret.Results.Topks = append(ret.Results.Topks, int64(len(groupList))-offset)
		for index := offset; index < int64(len(groupList)); index++ {
			group := groupList[index]
			typeutil.AppendPKs(ret.Results.Ids, group.idList)
			for _, score := range group.scoreList {
				if roundDecimal != -1 {
					multiplier := math.Pow(10.0, float64(roundDecimal))
					score = float32(math.Floor(float64(score)*multiplier+0.5) / multiplier)
				}
				ret.Results.Scores = append(ret.Results.Scores, score)
				typeutil.AppendGroupByValue(ret.Results, group.groupVal, groupByDataType)
			}
		}
	}

	return ret, nil

}

func rankSearchResultDataByPk(ctx context.Context,
	nq int64,
	params *rankParams,
	pkType schemapb.DataType,
	searchResults []*milvuspb.SearchResults,
) (*milvuspb.SearchResults, error) {
	tr := timerecord.NewTimeRecorder("rankSearchResultDataByPk")
	defer func() {
		tr.CtxElapse(ctx, "done")
	}()

	offset := params.offset
	limit := params.limit
	topk := limit + offset
	roundDecimal := params.roundDecimal
	log.Ctx(ctx).Debug("rankSearchResultDataByPk",
		zap.Int("len(searchResults)", len(searchResults)),
		zap.Int64("nq", nq),
		zap.Int64("offset", offset),
		zap.Int64("limit", limit))

	ret := &milvuspb.SearchResults{
		Status: merr.Success(),
		Results: &schemapb.SearchResultData{
			NumQueries: nq,
			TopK:       limit,
			FieldsData: make([]*schemapb.FieldData, 0),
			Scores:     []float32{},
			Ids:        &schemapb.IDs{},
			Topks:      []int64{},
		},
	}

	switch pkType {
	case schemapb.DataType_Int64:
		ret.GetResults().Ids.IdField = &schemapb.IDs_IntId{
			IntId: &schemapb.LongArray{
				Data: make([]int64, 0),
			},
		}
	case schemapb.DataType_VarChar:
		ret.GetResults().Ids.IdField = &schemapb.IDs_StrId{
			StrId: &schemapb.StringArray{
				Data: make([]string, 0),
			},
		}
	default:
		return nil, errors.New("unsupported pk type")
	}

	// []map[id]score
	accumulatedScores := make([]map[interface{}]float32, nq)
	for i := int64(0); i < nq; i++ {
		accumulatedScores[i] = make(map[interface{}]float32)
	}

	for _, result := range searchResults {
		scores := result.GetResults().GetScores()
		start := int64(0)
		for i := int64(0); i < nq; i++ {
			realTopk := result.GetResults().Topks[i]
			for j := start; j < start+realTopk; j++ {
				id := typeutil.GetPK(result.GetResults().GetIds(), j)
				accumulatedScores[i][id] += scores[j]
			}
			start += realTopk
		}
	}

	for i := int64(0); i < nq; i++ {
		idSet := accumulatedScores[i]
		keys := make([]interface{}, 0)
		for key := range idSet {
			keys = append(keys, key)
		}
		if int64(len(keys)) <= offset {
			ret.Results.Topks = append(ret.Results.Topks, 0)
			continue
		}

		compareKeys := func(keyI, keyJ interface{}) bool {
			switch keyI.(type) {
			case int64:
				return keyI.(int64) < keyJ.(int64)
			case string:
				return keyI.(string) < keyJ.(string)
			}
			return false
		}

		// sort id by score
		big := func(i, j int) bool {
			if idSet[keys[i]] == idSet[keys[j]] {
				return compareKeys(keys[i], keys[j])
			}
			return idSet[keys[i]] > idSet[keys[j]]
		}

		sort.Slice(keys, big)

		if int64(len(keys)) > topk {
			keys = keys[:topk]
		}

		// set real topk
		ret.Results.Topks = append(ret.Results.Topks, int64(len(keys))-offset)
		// append id and score
		for index := offset; index < int64(len(keys)); index++ {
			typeutil.AppendPKs(ret.Results.Ids, keys[index])
			score := idSet[keys[index]]
			if roundDecimal != -1 {
				multiplier := math.Pow(10.0, float64(roundDecimal))
				score = float32(math.Floor(float64(score)*multiplier+0.5) / multiplier)
			}
			ret.Results.Scores = append(ret.Results.Scores, score)
		}
	}

	return ret, nil
}

func fillInEmptyResult(numQueries int64) *milvuspb.SearchResults {
	return &milvuspb.SearchResults{
		Status: merr.Success("search result is empty"),
		Results: &schemapb.SearchResultData{
			NumQueries: numQueries,
			Topks:      make([]int64, numQueries),
		},
	}
}
