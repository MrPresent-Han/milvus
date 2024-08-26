package tasks

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/segcorepb"
	"github.com/milvus-io/milvus/internal/querynodev2/collector"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

var _ Task = &QueryTask{}

func NewQueryTask(ctx context.Context,
	collection *segments.Collection,
	manager *segments.Manager,
	req *querypb.QueryRequest,
) *QueryTask {
	ctx, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "schedule")
	return &QueryTask{
		ctx:            ctx,
		collection:     collection,
		segmentManager: manager,
		req:            req,
		notifier:       make(chan error, 1),
		tr:             timerecord.NewTimeRecorderWithTrace(ctx, "queryTask"),
		scheduleSpan:   span,
		isScanQuery:    req.Req.ScanCtx != nil,
	}
}

type QueryTask struct {
	ctx            context.Context
	collection     *segments.Collection
	segmentManager *segments.Manager
	req            *querypb.QueryRequest
	result         *internalpb.RetrieveResults
	notifier       chan error
	tr             *timerecord.TimeRecorder
	scheduleSpan   trace.Span
	isScanQuery    bool
}

// Return the username which task is belong to.
// Return "" if the task do not contain any user info.
func (t *QueryTask) Username() string {
	return t.req.Req.GetUsername()
}

func (t *QueryTask) IsGpuIndex() bool {
	return false
}

// PreExecute the task, only call once.
func (t *QueryTask) PreExecute() error {
	// Update task wait time metric before execute
	nodeID := strconv.FormatInt(paramtable.GetNodeID(), 10)
	inQueueDuration := t.tr.ElapseSpan()
	inQueueDurationMS := inQueueDuration.Seconds() * 1000

	// Update in queue metric for prometheus.
	metrics.QueryNodeSQLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.QueryLabel,
		t.collection.GetDBName(),
		t.collection.GetResourceGroup(), // TODO: resource group and db name may be removed at runtime.
		// should be refactor into metricsutil.observer in the future.
	).Observe(inQueueDurationMS)

	username := t.Username()
	metrics.QueryNodeSQPerUserLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.QueryLabel,
		username).
		Observe(inQueueDurationMS)

	// Update collector for query node quota.
	collector.Average.Add(metricsinfo.QueryQueueMetric, float64(inQueueDuration.Microseconds()))
	return nil
}

func (t *QueryTask) SearchResult() *internalpb.SearchResults {
	return nil
}

// Execute the task, only call once.
func (t *QueryTask) Execute() error {
	if t.scheduleSpan != nil {
		t.scheduleSpan.End()
	}
	tr := timerecord.NewTimeRecorderWithTrace(t.ctx, "QueryTask")

	retrievePlan, err := segments.NewRetrievePlan(
		t.ctx,
		t.collection,
		t.req.Req.GetSerializedExprPlan(),
		t.req.Req.GetMvccTimestamp(),
		t.req.Req.Base.GetMsgID(),
		t.req.Req.GetScanCtx(),
	)
	if err != nil {
		return err
	}
	defer retrievePlan.Delete()
	results, pinnedSegments, err := segments.Retrieve(t.ctx, t.segmentManager, retrievePlan, t.req)
	defer t.segmentManager.Segment.Unpin(pinnedSegments)
	if err != nil {
		return err
	}

	reducer := segments.CreateSegCoreReducer(
		t.req,
		t.collection.Schema(),
		t.segmentManager,
	)
	beforeReduce := time.Now()

	reduceResults := make([]*segcorepb.RetrieveResults, 0, len(results))
	querySegments := make([]segments.Segment, 0, len(results))
	for _, result := range results {
		reduceResults = append(reduceResults, result.Result)
		querySegments = append(querySegments, result.Segment)
	}
	reducedResult, err := reducer.Reduce(t.ctx, reduceResults, querySegments, retrievePlan)

	metrics.QueryNodeReduceLatency.WithLabelValues(
		fmt.Sprint(paramtable.GetNodeID()),
		metrics.QueryLabel,
		metrics.ReduceSegments,
		metrics.BatchReduce).Observe(float64(time.Since(beforeReduce).Milliseconds()))
	if err != nil {
		return err
	}

	relatedDataSize := lo.Reduce(querySegments, func(acc int64, seg segments.Segment, _ int) int64 {
		return acc + segments.GetSegmentRelatedDataSize(seg)
	}, 0)

	t.result = &internalpb.RetrieveResults{
		Base: &commonpb.MsgBase{
			SourceID: paramtable.GetNodeID(),
		},
		Status:     merr.Success(),
		Ids:        reducedResult.Ids,
		FieldsData: reducedResult.FieldsData,
		CostAggregation: &internalpb.CostAggregation{
			ServiceTime:          tr.ElapseSpan().Milliseconds(),
			TotalRelatedDataSize: relatedDataSize,
		},
		AllRetrieveCount: reducedResult.GetAllRetrieveCount(),
		HasMoreResult:    reducedResult.HasMoreResult,
	}
	t.handleScan(reducedResult)
	return nil
}

func (t *QueryTask) handleScan(reducedResult *segcorepb.RetrieveResults) {
	if t.req.GetReq().GetScanCtx() != nil {
		scanReqCtx := t.req.GetReq().GetScanCtx()
		var lastOffset int64 = -1
		offsetCount := len(reducedResult.GetOffset())
		if offsetCount > 0 {
			lastOffset = reducedResult.GetOffset()[offsetCount-1]
		}
		scanResCtx := &internalpb.ScanCtx{
			ScanCtxId:  scanReqCtx.GetScanCtxId(),
			SegmentIdx: scanReqCtx.GetSegmentIdx(),
			MvccTs:     scanReqCtx.GetMvccTs(),
			Offset:     lastOffset,
		}
		t.result.ScanCtx = scanResCtx
	}
}

func (t *QueryTask) Done(err error) {
	t.notifier <- err
}

func (t *QueryTask) Canceled() error {
	return t.ctx.Err()
}

func (t *QueryTask) Wait() error {
	return <-t.notifier
}

func (t *QueryTask) Result() *internalpb.RetrieveResults {
	return t.result
}

func (t *QueryTask) NQ() int64 {
	return 1
}
