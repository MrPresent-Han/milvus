package segments

import (
	"context"
	"fmt"
	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
)

func checkSearchResult(ctx context.Context, nq int64, plan *segments.SearchPlan, searchResult *segments.SearchResult) error {
	searchResults := make([]*segments.SearchResult, 0)
	searchResults = append(searchResults, searchResult)

	topK := plan.GetTopK()
	sliceNQs := []int64{nq / 5, nq / 5, nq / 5, nq / 5, nq / 5}
	sliceTopKs := []int64{topK, topK / 2, topK, topK, topK / 2}
	sInfo := ParseSliceInfo(sliceNQs, sliceTopKs, nq)

	res, err := ReduceSearchResultsAndFillData(ctx, plan, searchResults, 1, sInfo.SliceNQs, sInfo.SliceTopKs)
	if err != nil {
		return err
	}

	for i := 0; i < len(sInfo.SliceNQs); i++ {
		blob, err := GetSearchResultDataBlob(ctx, res, i)
		if err != nil {
			return err
		}
		if len(blob) == 0 {
			return fmt.Errorf("wrong search result data blobs when checkSearchResult")
		}

		result := &schemapb.SearchResultData{}
		err = proto.Unmarshal(blob, result)
		if err != nil {
			return err
		}

		if result.TopK != sliceTopKs[i] {
			return fmt.Errorf("unexpected topK when checkSearchResult")
		}
		if result.NumQueries != sInfo.SliceNQs[i] {
			return fmt.Errorf("unexpected nq when checkSearchResult")
		}
		// search empty segment, return empty result.IDs
		if len(result.Ids.IdField.(*schemapb.IDs_IntId).IntId.Data) <= 0 {
			return fmt.Errorf("unexpected Ids when checkSearchResult")
		}
		if len(result.Scores) <= 0 {
			return fmt.Errorf("unexpected Scores when checkSearchResult")
		}
	}

	segments.DeleteSearchResults(searchResults)
	segments.DeleteSearchResultDataBlobs(res)
	return nil
}
