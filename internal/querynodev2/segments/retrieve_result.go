package segments

import "C"

// RetrieveResult contains a pointer to the retrieve result in C++ memory
type RetrieveResult struct {
	cRetrieveResult C.CRetrieveResult
}
