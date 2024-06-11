package segments

import "C"

// SearchResult contains a pointer to the search result in C++ memory
type SearchResult struct {
	cSearchResult C.CSearchResult
}

func (result *SearchResult) GetCSearchResult() C.CSearchResult {
	return result.cSearchResult
}

// SearchResultDataBlobs is the CSearchResultsDataBlobs in C++
type SearchResultDataBlobs = C.CSearchResultDataBlobs

func DeleteSearchResults(results []*SearchResult) {
	if len(results) == 0 {
		return
	}
	for _, result := range results {
		if result != nil {
			C.DeleteSearchResult(result.cSearchResult)
		}
	}
}

func DeleteSearchResultDataBlobs(cSearchResultDataBlobs SearchResultDataBlobs) {
	C.DeleteSearchResultDataBlobs(cSearchResultDataBlobs)
}
