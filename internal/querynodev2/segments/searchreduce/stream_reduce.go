package segments

import "C"

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
)

type (
	StreamSearchReducer = C.CSearchStreamReducer
)

func NewStreamReducer(ctx context.Context,
	plan *segments.SearchPlan,
	sliceNQs []int64,
	sliceTopKs []int64,
) (StreamSearchReducer, error) {
	if plan.GetCSearchPlan() == nil {
		return nil, fmt.Errorf("nil search plan")
	}
	if len(sliceNQs) == 0 {
		return nil, fmt.Errorf("empty slice nqs is not allowed")
	}
	if len(sliceNQs) != len(sliceTopKs) {
		return nil, fmt.Errorf("unaligned sliceNQs(len=%d) and sliceTopKs(len=%d)", len(sliceNQs), len(sliceTopKs))
	}
	cSliceNQSPtr := (*C.int64_t)(&sliceNQs[0])
	cSliceTopKSPtr := (*C.int64_t)(&sliceTopKs[0])
	cNumSlices := C.int64_t(len(sliceNQs))

	var streamReducer StreamSearchReducer
	status := C.NewStreamReducer(plan.GetCSearchPlan(), cSliceNQSPtr, cSliceTopKSPtr, cNumSlices, &streamReducer)
	if err := segments.HandleCStatus(ctx, &status, "MergeSearchResultsWithOutputFields failed"); err != nil {
		return nil, err
	}
	return streamReducer, nil
}

func StreamReduceSearchResult(ctx context.Context,
	newResult *segments.SearchResult, streamReducer StreamSearchReducer,
) error {
	cSearchResults := make([]C.CSearchResult, 0)
	cSearchResults = append(cSearchResults, newResult.GetCSearchResult())
	cSearchResultPtr := &cSearchResults[0]

	status := C.StreamReduce(streamReducer, cSearchResultPtr, 1)
	if err := segments.HandleCStatus(ctx, &status, "StreamReduceSearchResult failed"); err != nil {
		return err
	}
	return nil
}

func GetStreamReduceResult(ctx context.Context, streamReducer StreamSearchReducer) (segments.SearchResultDataBlobs, error) {
	var cSearchResultDataBlobs segments.SearchResultDataBlobs
	status := C.GetStreamReduceResult(streamReducer, &cSearchResultDataBlobs)
	if err := segments.HandleCStatus(ctx, &status, "ReduceSearchResultsAndFillData failed"); err != nil {
		return nil, err
	}
	return cSearchResultDataBlobs, nil
}

func DeleteStreamReduceHelper(cStreamReduceHelper StreamSearchReducer) {
	C.DeleteStreamSearchReducer(cStreamReduceHelper)
}
