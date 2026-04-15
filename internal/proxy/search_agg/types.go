package search_agg

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
)

const (
	ScoreFieldID    int64 = -1
	CountAllFieldID int64 = 0
)

type SearchAggregationContext struct {
	NQ int64

	Levels []LevelContext

	// UserOutputFieldIDs records the fields the user explicitly requested in
	// output_fields. Populated by the caller after construction. Only these
	// appear in the final HitResult.Fields.
	UserOutputFieldIDs map[int64]struct{}

	// DerivedTopK is the number of distinct composite keys each NQ retains
	// downstream, equal to the product of every level's Size (empty / zero
	// levels treated as 1). Mirrors ES `terms.size` nested product.
	DerivedTopK int64

	// DerivedGroupSize is the per-composite-key doc retention budget. Equal to
	// the leaf level's TopHits.Size when set, otherwise 1 (ES-style TopHits
	// drives per-bucket doc count; absence means we just need a single
	// representative doc per group to power metric compute).
	DerivedGroupSize int64

	// groupByFieldIDs is the union of all Levels[*].OwnFieldIDs, derived at
	// construction time. Values for these fields are read from
	// SearchResultData.group_by_field_values (written by segcore's composite
	// group-by path).
	groupByFieldIDs map[int64]struct{}

	// allGroupByFieldIDsOrdered is the full composite-key field list in
	// OwnFieldIDs order across all levels; used by proxy-side cross-shard
	// groupReduce so the composite matches what delegator used.
	allGroupByFieldIDsOrdered []int64

	// extraOutputFieldIDs are non-group-by fields the proxy needs from
	// fields_data (metric sources, top_hits sort). They must be appended to
	// SearchRequest.OutputFieldsId by the caller so segcore writes them.
	// Stored sorted for deterministic downstream behavior.
	extraOutputFieldIDs []int64
}

// IsGroupByField reports whether fieldID appears in any level's OwnFieldIDs.
func (c *SearchAggregationContext) IsGroupByField(fieldID int64) bool {
	_, ok := c.groupByFieldIDs[fieldID]
	return ok
}

// ExtraOutputFieldIDs returns non-group-by field IDs the caller must append to
// SearchRequest.OutputFieldsId.
func (c *SearchAggregationContext) ExtraOutputFieldIDs() []int64 {
	return c.extraOutputFieldIDs
}

// AllGroupByFieldIDs returns the flattened composite-key field list in the
// order defined by each level's OwnFieldIDs. Used by proxy-side cross-shard
// groupReduce to build composite keys consistent with the delegator.
func (c *SearchAggregationContext) AllGroupByFieldIDs() []int64 {
	return c.allGroupByFieldIDsOrdered
}

type LevelContext struct {
	OwnFieldIDs []int64
	Metrics     map[string]MetricSpec
	// metricPlans mirror Metrics but pre-compile each alias into a concrete
	// agg.AggregateBase chain so computeLevel can delegate accumulation to
	// internal/agg rather than reimplement per-op logic.
	metricPlans []metricPlan
	TopHits     *TopHitsConfig
	Order       []OrderCriterion
	Size        int64
}

type MetricSpec struct {
	Op        string
	FieldID   int64
	FieldType schemapb.DataType
}

// metricPlan is the executable form of a MetricSpec: one alias maps to one or
// more agg.AggregateBase (avg expands into sum+count per internal/agg
// convention). The private slice ordering matches the init order in
// newBucketState so each FieldValue slot aligns 1:1 with an aggregate.
type metricPlan struct {
	alias      string
	spec       MetricSpec
	aggregates []agg.AggregateBase
}

type TopHitsConfig struct {
	Size int64
	Sort []SortCriterion
}

type SortCriterion struct {
	FieldID int64
	Dir     string
}

type OrderCriterion struct {
	Key string
	Dir string
}

type RowRef struct {
	ResultIdx int
	RowIdx    int
}

type AggBucketResult struct {
	Key   map[int64]any
	Count int64
	// Metrics holds finalized values from agg.FieldValue.Value(). Typed as
	// any so string min/max (and other non-numeric results) pass through
	// without a lossy float64 coercion.
	Metrics   map[string]any
	Hits      []*HitResult
	SubGroups []*AggBucketResult
}

type HitResult struct {
	PK     any
	Score  float32
	Fields map[int64]any
}
