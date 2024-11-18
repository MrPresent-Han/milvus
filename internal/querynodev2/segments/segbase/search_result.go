package segbase

/*
#cgo pkg-config: milvus_core

#include "segcore/plan_c.h"
#include "segcore/reduce_c.h"
*/
import "C"

// SearchResult contains a pointer to the search result in C++ memory
type SearchResult struct {
	CSearchResult C.CSearchResult
}

func DeleteSearchResults(results []*SearchResult) {
	if len(results) == 0 {
		return
	}
	for _, result := range results {
		if result != nil {
			C.DeleteSearchResult(result.CSearchResult)
		}
	}
}
