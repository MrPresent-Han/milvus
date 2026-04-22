package search_agg

import (
	"fmt"
	"sort"
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/agg"
	typeutil2 "github.com/milvus-io/milvus/internal/util/typeutil"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

func BuildSearchAggregationContext(
	groupBy *commonpb.SearchAggregationSpec,
	schema *schemapb.CollectionSchema,
	nq int64,
) (*SearchAggregationContext, error) {
	resolved, err := resolveAggregationSpec(groupBy, schema)
	if err != nil {
		return nil, err
	}
	return NewContext(nq, resolved.levels, nil, resolved.extraOutputFieldIDs), nil
}

// NewContext assembles a SearchAggregationContext from already-resolved levels.
// Used by BuildSearchAggregationContext and by unit tests that want to drive
// the computer without parsing a real SearchAggregationSpec.
//
// groupByFieldIDs is derived from levels.OwnFieldIDs; extraOutputFieldIDs is
// sanitized (group-by fields removed) and sorted for deterministic ordering.
func NewContext(
	nq int64,
	levels []LevelContext,
	userOutputFieldIDs []int64,
	extraOutputFieldIDs []int64,
) *SearchAggregationContext {
	groupBy := make(map[int64]struct{})
	allGroupBy := make([]int64, 0)
	for _, level := range levels {
		for _, id := range level.OwnFieldIDs {
			if _, dup := groupBy[id]; dup {
				continue
			}
			groupBy[id] = struct{}{}
			allGroupBy = append(allGroupBy, id)
		}
	}

	extra := make([]int64, 0, len(extraOutputFieldIDs))
	seen := make(map[int64]struct{}, len(extraOutputFieldIDs))
	for _, id := range extraOutputFieldIDs {
		if _, isGroupBy := groupBy[id]; isGroupBy {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		extra = append(extra, id)
	}
	sort.Slice(extra, func(i, j int) bool { return extra[i] < extra[j] })

	userOutput := make(map[int64]struct{}, len(userOutputFieldIDs))
	for _, id := range userOutputFieldIDs {
		userOutput[id] = struct{}{}
	}

	topK, groupSize := deriveTopKAndGroupSize(levels)

	// Compile metricPlans for any level where the caller populated Metrics
	// but not metricPlans (typical for BuildSearchAggregationContext —
	// which already assembled plans — and for tests that handcraft Metrics).
	// Panic on compile failure: invalid metric specs are a caller-side bug.
	for i := range levels {
		if len(levels[i].Metrics) == 0 || len(levels[i].metricPlans) > 0 {
			continue
		}
		plans, err := compileMetricPlans(levels[i].Metrics)
		if err != nil {
			panic(fmt.Errorf("search_agg.NewContext: level %d metric compile failed: %w", i, err))
		}
		levels[i].metricPlans = plans
	}

	return &SearchAggregationContext{
		NQ:                        nq,
		Levels:                    levels,
		UserOutputFieldIDs:        userOutput,
		DerivedTopK:               topK,
		DerivedGroupSize:          groupSize,
		groupByFieldIDs:           groupBy,
		allGroupByFieldIDsOrdered: allGroupBy,
		extraOutputFieldIDs:       extra,
	}
}

// compileMetricPlans turns user-populated Metrics into the executable plan
// form used by the computer. Shared between BuildSearchAggregationContext and
// tests that bypass spec parsing.
func compileMetricPlans(metrics map[string]MetricSpec) ([]metricPlan, error) {
	aliases := make([]string, 0, len(metrics))
	for alias := range metrics {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	plans := make([]metricPlan, 0, len(aliases))
	for _, alias := range aliases {
		spec := metrics[alias]
		aggs, err := agg.NewAggregate(spec.Op, spec.FieldID, alias, spec.FieldType)
		if err != nil {
			return nil, fmt.Errorf("metric %q: %w", alias, err)
		}
		plans = append(plans, metricPlan{
			alias:      alias,
			spec:       spec,
			aggregates: aggs,
		})
	}
	return plans, nil
}

// deriveTopKAndGroupSize maps the ES-style nested aggregation sizes into the
// (topK, groupSize) pair consumed by the Delegator/QN group-reduce algorithm:
//
//	topK      = product of every level's Size (distinct composite keys cap)
//	groupSize = leaf level's TopHits.Size, or 1 when TopHits is absent
//
// See discussion tying `terms.size` and `top_hits.size` semantics.
func deriveTopKAndGroupSize(levels []LevelContext) (topK, groupSize int64) {
	topK = 1
	for _, lvl := range levels {
		if lvl.Size > 0 {
			topK *= lvl.Size
		}
	}
	groupSize = 1
	if n := len(levels); n > 0 {
		if leaf := levels[n-1]; leaf.TopHits != nil && leaf.TopHits.Size > 0 {
			groupSize = leaf.TopHits.Size
		}
	}
	return topK, groupSize
}

type resolvedAggregationSpec struct {
	levels          []LevelContext
	groupByFieldIDs []int64
	// extraOutputFieldIDs are non-group-by fields proxy needs from fields_data:
	// metric source fields and top_hits sort fields. These must be appended to
	// SearchRequest.OutputFieldsId so segcore writes them into fields_data.
	extraOutputFieldIDs []int64
}

func resolveAggregationSpec(groupBy *commonpb.SearchAggregationSpec, schema *schemapb.CollectionSchema) (*resolvedAggregationSpec, error) {
	if groupBy == nil {
		return nil, fmt.Errorf("group_by spec is nil")
	}
	if schema == nil {
		return nil, fmt.Errorf("collection schema is nil")
	}

	dynamicField := findDynamicField(schema)
	groupBySeen := make(map[int64]struct{})
	extraSeen := make(map[int64]struct{})

	resolved := &resolvedAggregationSpec{}
	var walk func(spec *commonpb.SearchAggregationSpec) error
	walk = func(spec *commonpb.SearchAggregationSpec) error {
		if spec == nil {
			return nil
		}

		if len(spec.GetFields()) == 0 {
			return fmt.Errorf("group_by level has no fields")
		}

		level := LevelContext{
			OwnFieldIDs: make([]int64, 0, len(spec.GetFields())),
			Size:        spec.GetSize(),
		}

		levelFieldSeen := make(map[int64]struct{})
		for _, fieldName := range spec.GetFields() {
			fieldID, err := resolveFieldID(fieldName, schema, dynamicField)
			if err != nil {
				return fmt.Errorf("invalid group_by field %q: %w", fieldName, err)
			}
			if _, ok := levelFieldSeen[fieldID]; ok {
				return fmt.Errorf("duplicated group_by field %q in one level", fieldName)
			}
			if _, ok := groupBySeen[fieldID]; ok {
				return fmt.Errorf("duplicated group_by field %q across levels", fieldName)
			}
			levelFieldSeen[fieldID] = struct{}{}
			groupBySeen[fieldID] = struct{}{}
			level.OwnFieldIDs = append(level.OwnFieldIDs, fieldID)
			resolved.groupByFieldIDs = append(resolved.groupByFieldIDs, fieldID)
		}

		if len(spec.GetMetrics()) > 0 {
			level.Metrics = make(map[string]MetricSpec, len(spec.GetMetrics()))
			// Sort aliases so metricPlans order is deterministic regardless of
			// proto map iteration order — matters for reproducible finalization.
			aliases := make([]string, 0, len(spec.GetMetrics()))
			for alias := range spec.GetMetrics() {
				aliases = append(aliases, alias)
			}
			sort.Strings(aliases)
			level.metricPlans = make([]metricPlan, 0, len(aliases))
			for _, alias := range aliases {
				metric := spec.GetMetrics()[alias]
				if strings.TrimSpace(alias) == "" {
					return fmt.Errorf("metric alias cannot be empty")
				}
				metricSpec, metricSourceFieldID, err := buildMetricSpec(metric, schema, dynamicField)
				if err != nil {
					return fmt.Errorf("invalid metric %q: %w", alias, err)
				}
				aggs, err := agg.NewAggregate(metricSpec.Op, metricSpec.FieldID, alias, metricSpec.FieldType)
				if err != nil {
					return fmt.Errorf("invalid metric %q: %w", alias, err)
				}
				level.Metrics[alias] = metricSpec
				level.metricPlans = append(level.metricPlans, metricPlan{
					alias:      alias,
					spec:       metricSpec,
					aggregates: aggs,
				})
				appendUniqueFieldID(extraSeen, &resolved.extraOutputFieldIDs, metricSourceFieldID)
			}
		}

		topHits, topHitsFieldIDs, err := buildTopHitsConfig(spec.GetTopHits(), schema, dynamicField)
		if err != nil {
			return err
		}
		level.TopHits = topHits
		for _, fieldID := range topHitsFieldIDs {
			appendUniqueFieldID(extraSeen, &resolved.extraOutputFieldIDs, fieldID)
		}

		order, err := buildOrderCriteria(spec.GetOrder(), level.Metrics)
		if err != nil {
			return err
		}
		level.Order = order

		resolved.levels = append(resolved.levels, level)
		return walk(spec.GetSubAggregation())
	}

	if err := walk(groupBy); err != nil {
		return nil, err
	}

	return resolved, nil
}

func buildMetricSpec(metric *commonpb.MetricAggSpec, schema *schemapb.CollectionSchema, dynamicField *schemapb.FieldSchema) (MetricSpec, int64, error) {
	if metric == nil {
		return MetricSpec{}, 0, fmt.Errorf("metric spec is nil")
	}

	op := strings.ToLower(strings.TrimSpace(metric.GetOp()))
	switch op {
	case "avg", "sum", "count", "min", "max":
	default:
		return MetricSpec{}, 0, fmt.Errorf("unsupported metric op %q", metric.GetOp())
	}

	fieldName := strings.TrimSpace(metric.GetFieldName())
	if fieldName == "" {
		return MetricSpec{}, 0, fmt.Errorf("metric field_name is empty")
	}

	switch fieldName {
	case "_score":
		// _score is a float32 produced by the engine; declare as Float so
		// internal/agg's type check accepts it for sum/avg/min/max.
		return MetricSpec{Op: op, FieldID: ScoreFieldID, FieldType: schemapb.DataType_Float}, 0, nil
	case "*":
		if op != "count" {
			return MetricSpec{}, 0, fmt.Errorf("field_name '*' only supports count op")
		}
		// count(*) has no source field; DataType_None is what internal/agg
		// expects for the synthetic always-1 input.
		return MetricSpec{Op: op, FieldID: CountAllFieldID, FieldType: schemapb.DataType_None}, 0, nil
	default:
		fieldID, err := resolveFieldID(fieldName, schema, dynamicField)
		if err != nil {
			return MetricSpec{}, 0, err
		}
		fieldType := schemapb.DataType_None
		if field := typeutil.GetFieldByName(schema, fieldName); field != nil {
			fieldType = field.GetDataType()
		}
		return MetricSpec{Op: op, FieldID: fieldID, FieldType: fieldType}, fieldID, nil
	}
}

func buildTopHitsConfig(topHits *commonpb.TopHitsSpec, schema *schemapb.CollectionSchema, dynamicField *schemapb.FieldSchema) (*TopHitsConfig, []int64, error) {
	if topHits == nil {
		return nil, nil, nil
	}

	cfg := &TopHitsConfig{
		Size: topHits.GetSize(),
		Sort: make([]SortCriterion, 0, len(topHits.GetSort())),
	}
	sortFieldIDs := make([]int64, 0, len(topHits.GetSort()))
	seen := make(map[int64]struct{})

	for _, sortSpec := range topHits.GetSort() {
		if sortSpec == nil {
			return nil, nil, fmt.Errorf("top_hits.sort contains nil item")
		}
		fieldName := strings.TrimSpace(sortSpec.GetFieldName())
		if fieldName == "" {
			return nil, nil, fmt.Errorf("top_hits.sort field_name is empty")
		}

		direction, err := normalizeDirection(sortSpec.GetDirection(), "desc")
		if err != nil {
			return nil, nil, fmt.Errorf("invalid top_hits.sort direction for %q: %w", fieldName, err)
		}

		if fieldName == "_score" {
			cfg.Sort = append(cfg.Sort, SortCriterion{FieldID: ScoreFieldID, Dir: direction})
			continue
		}

		fieldID, err := resolveFieldID(fieldName, schema, dynamicField)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid top_hits.sort field %q: %w", fieldName, err)
		}
		cfg.Sort = append(cfg.Sort, SortCriterion{FieldID: fieldID, Dir: direction})
		appendUniqueFieldID(seen, &sortFieldIDs, fieldID)
	}

	return cfg, sortFieldIDs, nil
}

func buildOrderCriteria(orderSpecs []*commonpb.OrderSpec, metrics map[string]MetricSpec) ([]OrderCriterion, error) {
	if len(orderSpecs) == 0 {
		return nil, nil
	}

	order := make([]OrderCriterion, 0, len(orderSpecs))
	for _, orderSpec := range orderSpecs {
		if orderSpec == nil {
			return nil, fmt.Errorf("order contains nil item")
		}

		key := strings.TrimSpace(orderSpec.GetKey())
		if key == "" {
			return nil, fmt.Errorf("order key is empty")
		}

		if key != "_count" && key != "_key" {
			if _, ok := metrics[key]; !ok {
				return nil, fmt.Errorf("order key %q is neither reserved key nor metric alias", key)
			}
		}

		direction, err := normalizeDirection(orderSpec.GetDirection(), "desc")
		if err != nil {
			return nil, fmt.Errorf("invalid order direction for key %q: %w", key, err)
		}
		order = append(order, OrderCriterion{Key: key, Dir: direction})
	}

	return order, nil
}

func normalizeDirection(direction string, defaultDir string) (string, error) {
	dir := strings.ToLower(strings.TrimSpace(direction))
	if dir == "" {
		return defaultDir, nil
	}
	switch dir {
	case "asc", "desc":
		return dir, nil
	default:
		return "", fmt.Errorf("direction must be asc or desc")
	}
}

func resolveFieldID(fieldExpr string, schema *schemapb.CollectionSchema, dynamicField *schemapb.FieldSchema) (int64, error) {
	fieldExpr = strings.TrimSpace(fieldExpr)
	if fieldExpr == "" {
		return 0, fmt.Errorf("field is empty")
	}

	if field := typeutil.GetFieldByName(schema, fieldExpr); field != nil {
		return field.GetFieldID(), nil
	}

	hasBrackets := strings.Contains(fieldExpr, "[") && strings.Contains(fieldExpr, "]")
	if hasBrackets {
		baseName := strings.Split(fieldExpr, "[")[0]
		if baseName != "" {
			if baseField := typeutil.GetFieldByName(schema, baseName); baseField != nil {
				if _, err := typeutil2.ParseAndVerifyNestedPath(fieldExpr, schema, baseField.GetFieldID()); err != nil {
					return 0, err
				}
				return baseField.GetFieldID(), nil
			}
		}

		if dynamicField != nil {
			if _, err := typeutil2.ParseAndVerifyNestedPath(fieldExpr, schema, dynamicField.GetFieldID()); err != nil {
				return 0, err
			}
			return dynamicField.GetFieldID(), nil
		}
	}

	if dynamicField != nil {
		if _, err := typeutil2.ParseAndVerifyNestedPath(fieldExpr, schema, dynamicField.GetFieldID()); err == nil {
			return dynamicField.GetFieldID(), nil
		}
	}

	return 0, fmt.Errorf("field %q not found in schema", fieldExpr)
}

func findDynamicField(schema *schemapb.CollectionSchema) *schemapb.FieldSchema {
	for _, field := range typeutil.GetAllFieldSchemas(schema) {
		if field.GetIsDynamic() {
			return field
		}
	}
	return nil
}

func appendUniqueFieldID(seen map[int64]struct{}, fields *[]int64, fieldID int64) {
	if fieldID <= 0 {
		return
	}
	if _, ok := seen[fieldID]; ok {
		return
	}
	seen[fieldID] = struct{}{}
	*fields = append(*fields, fieldID)
}
