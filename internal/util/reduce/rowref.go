package reduce

// RowRef references row RowIdx within the ResultIdx-th sub-result of a
// multi-source SearchResultData / RetrieveResults slice. It is the zero-copy
// pointer used by group-by reducers to track which surviving rows to emit,
// avoiding the cost of materializing copies during the merge walk.
type RowRef struct {
	ResultIdx int
	RowIdx    int64
}
