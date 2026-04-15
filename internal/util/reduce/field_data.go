package reduce

import (
	"fmt"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// FindFieldDataByID returns the first FieldData whose FieldId matches fieldID,
// or nil if none. Callers rely on FieldId being set upstream; when reducers
// re-emit group-by columns they must preserve FieldId for downstream lookup.
func FindFieldDataByID(fieldsData []*schemapb.FieldData, fieldID int64) *schemapb.FieldData {
	for _, fd := range fieldsData {
		if fd.GetFieldId() == fieldID {
			return fd
		}
	}
	return nil
}

// WriteGroupByFieldValues emits one FieldData per composite-key field into
// ret.GroupByFieldValues, pulling values for each accepted row from the
// source shard it originated in. Each emitted FieldData carries FieldId so
// downstream consumers can look up by id rather than position.
//
// Shared by delegator-side SearchGroupByReduce and proxy-side cross-shard
// reduce, so both layers produce a byte-for-byte identical field-17 payload.
func WriteGroupByFieldValues(
	ret *schemapb.SearchResultData,
	acceptedRows []RowRef,
	sources []*schemapb.SearchResultData,
	fieldIDs []int64,
) error {
	if len(fieldIDs) == 0 {
		return nil
	}
	ret.GroupByFieldValues = make([]*schemapb.FieldData, 0, len(fieldIDs))
	for _, fid := range fieldIDs {
		iters := make([]func(int) any, len(sources))
		var template *schemapb.FieldData
		for i, srd := range sources {
			fd := FindFieldDataByID(srd.GetGroupByFieldValues(), fid)
			if fd == nil {
				continue
			}
			iters[i] = typeutil.GetDataIterator(fd)
			if template == nil {
				template = fd
			}
		}
		if template == nil {
			return fmt.Errorf("group-by field %d not present in any source's group_by_field_values", fid)
		}

		builder, err := typeutil.NewFieldDataBuilder(template.GetType(), true, len(acceptedRows))
		if err != nil {
			return err
		}
		for _, row := range acceptedRows {
			iter := iters[row.ResultIdx]
			if iter == nil {
				return fmt.Errorf("group-by field %d missing at source index %d", fid, row.ResultIdx)
			}
			builder.Add(iter(int(row.RowIdx)))
		}
		fd := builder.Build()
		fd.FieldId = fid
		ret.GroupByFieldValues = append(ret.GroupByFieldValues, fd)
	}
	return nil
}
