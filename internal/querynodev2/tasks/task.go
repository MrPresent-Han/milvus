package tasks

// TODO: rename this file into search_task.go

import "C"

import (
	"bytes"
	"context"
	"fmt"
	"strconv"

	"github.com/golang/protobuf/proto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querynodev2/collector"
	"github.com/milvus-io/milvus/internal/querynodev2/segments"
	"github.com/milvus-io/milvus/pkg/log"
	"github.com/milvus-io/milvus/pkg/metrics"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"github.com/milvus-io/milvus/pkg/util/merr"
	"github.com/milvus-io/milvus/pkg/util/metricsinfo"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/milvus-io/milvus/pkg/util/timerecord"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

var (
	_ Task      = &SearchTask{}
	_ MergeTask = &SearchTask{}
)

type SearchTask struct {
	ctx              context.Context
	collection       *segments.Collection
	segmentManager   *segments.Manager
	req              *querypb.SearchRequest
	result           *internalpb.SearchResults
	merged           bool
	groupSize        int64
	topk             int64
	nq               int64
	placeholderGroup []byte
	originTopks      []int64
	originNqs        []int64
	others           []*SearchTask
	notifier         chan error
	serverID         int64

	tr           *timerecord.TimeRecorder
	scheduleSpan trace.Span
}

func NewSearchTask(ctx context.Context,
	collection *segments.Collection,
	manager *segments.Manager,
	req *querypb.SearchRequest,
	serverID int64,
) *SearchTask {
	ctx, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "schedule")
	return &SearchTask{
		ctx:              ctx,
		collection:       collection,
		segmentManager:   manager,
		req:              req,
		merged:           false,
		groupSize:        1,
		topk:             req.GetReq().GetTopk(),
		nq:               req.GetReq().GetNq(),
		placeholderGroup: req.GetReq().GetPlaceholderGroup(),
		originTopks:      []int64{req.GetReq().GetTopk()},
		originNqs:        []int64{req.GetReq().GetNq()},
		notifier:         make(chan error, 1),
		tr:               timerecord.NewTimeRecorderWithTrace(ctx, "searchTask"),
		scheduleSpan:     span,
		serverID:         serverID,
	}
}

// Return the username which task is belong to.
// Return "" if the task do not contain any user info.
func (t *SearchTask) Username() string {
	return t.req.Req.GetUsername()
}

func (t *SearchTask) GetNodeID() int64 {
	return t.serverID
}

func (t *SearchTask) IsGpuIndex() bool {
	return t.collection.IsGpuIndex()
}

func (t *SearchTask) PreExecute() error {
	// Update task wait time metric before execute
	nodeID := strconv.FormatInt(t.GetNodeID(), 10)
	inQueueDuration := t.tr.ElapseSpan()

	// Update in queue metric for prometheus.
	metrics.QueryNodeSQLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel).
		Observe(float64(inQueueDuration.Milliseconds()))

	username := t.Username()
	metrics.QueryNodeSQPerUserLatencyInQueue.WithLabelValues(
		nodeID,
		metrics.SearchLabel,
		username).
		Observe(float64(inQueueDuration.Milliseconds()))

	// Update collector for query node quota.
	collector.Average.Add(metricsinfo.SearchQueueMetric, float64(inQueueDuration.Microseconds()))

	// Execute merged task's PreExecute.
	for _, subTask := range t.others {
		err := subTask.PreExecute()
		if err != nil {
			return err
		}
	}
	return nil
}

func (t *SearchTask) Execute() error {
	log := log.Ctx(t.ctx).With(
		zap.Int64("collectionID", t.collection.ID()),
		zap.String("shard", t.req.GetDmlChannels()[0]),
	)

	if t.scheduleSpan != nil {
		t.scheduleSpan.End()
	}
	tr := timerecord.NewTimeRecorderWithTrace(t.ctx, "SearchTask")

	req := t.req
	t.combinePlaceHolderGroups()
	searchReq, err := segments.NewSearchRequest(t.ctx, t.collection, req, t.placeholderGroup)
	if err != nil {
		return err
	}
	defer searchReq.Delete()

	var (
		results          []*segments.SearchResult
		searchedSegments []segments.Segment
	)
	if req.GetScope() == querypb.DataScope_Historical {
		results, searchedSegments, err = segments.SearchHistorical(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			nil,
			req.GetSegmentIDs(),
		)
	} else if req.GetScope() == querypb.DataScope_Streaming {
		results, searchedSegments, err = segments.SearchStreaming(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			nil,
			req.GetSegmentIDs(),
		)
	}
	defer t.segmentManager.Segment.Unpin(searchedSegments)
	if err != nil {
		return err
	}
	defer segments.DeleteSearchResults(results)

	// plan.MetricType is accurate, though req.MetricType may be empty
	metricType := searchReq.Plan().GetMetricType()

	if len(results) == 0 {
		for i := range t.originNqs {
			var task *SearchTask
			if i == 0 {
				task = t
			} else {
				task = t.others[i-1]
			}

			task.result = &internalpb.SearchResults{
				Base: &commonpb.MsgBase{
					SourceID: t.GetNodeID(),
				},
				Status:         merr.Success(),
				MetricType:     metricType,
				NumQueries:     t.originNqs[i],
				TopK:           t.originTopks[i],
				SlicedOffset:   1,
				SlicedNumCount: 1,
				CostAggregation: &internalpb.CostAggregation{
					ServiceTime: tr.ElapseSpan().Milliseconds(),
				},
			}
		}
		return nil
	}

	tr.RecordSpan()
	blobs, err := segments.ReduceSearchResultsAndFillData(
		t.ctx,
		searchReq.Plan(),
		results,
		int64(len(results)),
		t.originNqs,
		t.originTopks,
	)
	if err != nil {
		log.Warn("failed to reduce search results", zap.Error(err))
		return err
	}
	defer segments.DeleteSearchResultDataBlobs(blobs)
	metrics.QueryNodeReduceLatency.WithLabelValues(
		fmt.Sprint(t.GetNodeID()),
		metrics.SearchLabel,
		metrics.ReduceSegments).
		Observe(float64(tr.RecordSpan().Milliseconds()))
	for i := range t.originNqs {
		blob, err := segments.GetSearchResultDataBlob(t.ctx, blobs, i)
		if err != nil {
			return err
		}

		var task *SearchTask
		if i == 0 {
			task = t
		} else {
			task = t.others[i-1]
		}

		// Note: blob is unsafe because get from C
		bs := make([]byte, len(blob))
		copy(bs, blob)

		task.result = &internalpb.SearchResults{
			Base: &commonpb.MsgBase{
				SourceID: t.GetNodeID(),
			},
			Status:         merr.Success(),
			MetricType:     metricType,
			NumQueries:     t.originNqs[i],
			TopK:           t.originTopks[i],
			SlicedBlob:     bs,
			SlicedOffset:   1,
			SlicedNumCount: 1,
			CostAggregation: &internalpb.CostAggregation{
				ServiceTime: tr.ElapseSpan().Milliseconds(),
			},
		}
	}

	return nil
}

func (t *SearchTask) Merge(other *SearchTask) bool {
	var (
		nq        = t.nq
		topk      = t.topk
		otherNq   = other.nq
		otherTopk = other.topk
	)

	diffTopk := topk != otherTopk
	pre := funcutil.Min(nq*topk, otherNq*otherTopk)
	maxTopk := funcutil.Max(topk, otherTopk)
	after := (nq + otherNq) * maxTopk
	ratio := float64(after) / float64(pre)

	// Check mergeable
	if t.req.GetReq().GetDbID() != other.req.GetReq().GetDbID() ||
		t.req.GetReq().GetCollectionID() != other.req.GetReq().GetCollectionID() ||
		t.req.GetReq().GetMvccTimestamp() != other.req.GetReq().GetMvccTimestamp() ||
		t.req.GetReq().GetDslType() != other.req.GetReq().GetDslType() ||
		t.req.GetDmlChannels()[0] != other.req.GetDmlChannels()[0] ||
		nq+otherNq > paramtable.Get().QueryNodeCfg.MaxGroupNQ.GetAsInt64() ||
		diffTopk && ratio > paramtable.Get().QueryNodeCfg.TopKMergeRatio.GetAsFloat() ||
		!funcutil.SliceSetEqual(t.req.GetReq().GetPartitionIDs(), other.req.GetReq().GetPartitionIDs()) ||
		!funcutil.SliceSetEqual(t.req.GetSegmentIDs(), other.req.GetSegmentIDs()) ||
		!bytes.Equal(t.req.GetReq().GetSerializedExprPlan(), other.req.GetReq().GetSerializedExprPlan()) {
		return false
	}

	// Merge
	t.groupSize += other.groupSize
	t.topk = maxTopk
	t.nq += otherNq
	t.originTopks = append(t.originTopks, other.originTopks...)
	t.originNqs = append(t.originNqs, other.originNqs...)
	t.others = append(t.others, other)
	other.merged = true

	return true
}

func (t *SearchTask) Done(err error) {
	if !t.merged {
		metrics.QueryNodeSearchGroupSize.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.groupSize))
		metrics.QueryNodeSearchGroupNQ.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.nq))
		metrics.QueryNodeSearchGroupTopK.WithLabelValues(fmt.Sprint(t.GetNodeID())).Observe(float64(t.topk))
	}
	t.notifier <- err
	for _, other := range t.others {
		other.Done(err)
	}
}

func (t *SearchTask) Canceled() error {
	return t.ctx.Err()
}

func (t *SearchTask) Wait() error {
	return <-t.notifier
}

func (t *SearchTask) SearchResult() *internalpb.SearchResults {
	if t.result != nil {
		channelsMvcc := make(map[string]uint64)
		for _, ch := range t.req.GetDmlChannels() {
			channelsMvcc[ch] = t.req.GetReq().GetMvccTimestamp()
		}
		t.result.ChannelsMvcc = channelsMvcc
	}
	return t.result
}

func (t *SearchTask) NQ() int64 {
	return t.nq
}

func (t *SearchTask) MergeWith(other Task) bool {
	switch other := other.(type) {
	case *SearchTask:
		return t.Merge(other)
	}
	return false
}

// combinePlaceHolderGroups combine all the placeholder groups.
func (t *SearchTask) combinePlaceHolderGroups() {
	if len(t.others) > 0 {
		ret := &commonpb.PlaceholderGroup{}
		_ = proto.Unmarshal(t.placeholderGroup, ret)
		for _, t := range t.others {
			x := &commonpb.PlaceholderGroup{}
			_ = proto.Unmarshal(t.placeholderGroup, x)
			ret.Placeholders[0].Values = append(ret.Placeholders[0].Values, x.Placeholders[0].Values...)
		}
		t.placeholderGroup, _ = proto.Marshal(ret)
	}
}

type StreamingSearchTask struct {
	SearchTask
	others        []*StreamingSearchTask
	resultBlobs   segments.SearchResultDataBlobs
	streamReducer segments.StreamSearchReducer
}

func NewStreamingSearchTask(ctx context.Context,
	collection *segments.Collection,
	manager *segments.Manager,
	req *querypb.SearchRequest,
	serverID int64,
) *StreamingSearchTask {
	ctx, span := otel.Tracer(typeutil.QueryNodeRole).Start(ctx, "schedule")
	return &StreamingSearchTask{
		SearchTask: SearchTask{
			ctx:              ctx,
			collection:       collection,
			segmentManager:   manager,
			req:              req,
			merged:           false,
			groupSize:        1,
			topk:             req.GetReq().GetTopk(),
			nq:               req.GetReq().GetNq(),
			placeholderGroup: req.GetReq().GetPlaceholderGroup(),
			originTopks:      []int64{req.GetReq().GetTopk()},
			originNqs:        []int64{req.GetReq().GetNq()},
			notifier:         make(chan error, 1),
			tr:               timerecord.NewTimeRecorderWithTrace(ctx, "searchTask"),
			scheduleSpan:     span,
			serverID:         serverID,
		},
	}
}

func (t *StreamingSearchTask) MergeWith(other Task) bool {
	return false
}

func (t *StreamingSearchTask) Execute() error {
	log := log.Ctx(t.ctx).With(
		zap.Int64("collectionID", t.collection.ID()),
		zap.String("shard", t.req.GetDmlChannels()[0]),
	)
	// 0. prepare search req
	if t.scheduleSpan != nil {
		t.scheduleSpan.End()
	}
	tr := timerecord.NewTimeRecorderWithTrace(t.ctx, "SearchTask")
	req := t.req
	t.combinePlaceHolderGroups()
	searchReq, err := segments.NewSearchRequest(t.ctx, t.collection, req, t.placeholderGroup)
	if err != nil {
		return err
	}
	defer searchReq.Delete()

	var pinnedSegments []segments.Segment
	// 1. search&&reduce or streaming-search&&streaming-reduce
	metricType := searchReq.Plan().GetMetricType()
	if req.GetScope() == querypb.DataScope_Historical {
		streamingResultsChan := make(chan *segments.SearchResult, len(req.SegmentIDs))
		errStream := make(chan error, len(req.SegmentIDs))
		pinnedSegments, err = segments.SearchHistoricalStreamly(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			nil,
			req.GetSegmentIDs(),
			streamingResultsChan,
			errStream)
		if err != nil {
			log.Error("Failed to search sealed segments streamly", zap.Error(err))
			return err
		}
		searchResultsToDelete := make([]*segments.SearchResult, 0)
		var searchErr error
		var reduceErr error
		for result := range streamingResultsChan {
			searchResultsToDelete = append(searchResultsToDelete, result)
			log.Debug("streamingResultChan got streamed result")
			searchErr = <-errStream
			if searchErr != nil {
				break
			}
			log.Debug("streamingResultChan before doing stream reduce")
			reduceErr = t.streamReduce(t.ctx, searchReq.Plan(), result, t.originNqs, t.originTopks)
			log.Debug("streamingResultChan after doing stream reduce", zap.Error(reduceErr))
			if reduceErr != nil {
				break
			}
		}
		defer segments.DeleteStreamReduceHelper(t.streamReducer)
		defer segments.DeleteSearchResults(searchResultsToDelete)
		if searchErr != nil {
			log.Error("Failed to get search result from segments", zap.Error(searchErr))
			return searchErr
		}
		if reduceErr != nil {
			log.Error("Failed to stream reduce searched segments", zap.Error(reduceErr))
			return reduceErr
		}
		t.resultBlobs, err = segments.GetStreamReduceResult(t.ctx, t.streamReducer)
		defer segments.DeleteSearchResultDataBlobs(t.resultBlobs)
		if err != nil {
			log.Error("Failed to get stream-reduced search result")
			return err
		}
	} else if req.GetScope() == querypb.DataScope_Streaming {
		var results []*segments.SearchResult
		results, pinnedSegments, err = segments.SearchStreaming(
			t.ctx,
			t.segmentManager,
			searchReq,
			req.GetReq().GetCollectionID(),
			nil,
			req.GetSegmentIDs(),
		)
		defer segments.DeleteSearchResults(results)
		if err != nil {
			return err
		}
		if t.maybeReturnForEmptyResults(results, metricType, tr) {
			return nil
		}
		tr.RecordSpan()
		t.resultBlobs, err = segments.ReduceSearchResultsAndFillData(
			t.ctx,
			searchReq.Plan(),
			results,
			int64(len(results)),
			t.originNqs,
			t.originTopks,
		)
		if err != nil {
			log.Warn("failed to reduce search results", zap.Error(err))
			return err
		}
		defer segments.DeleteSearchResultDataBlobs(t.resultBlobs)
		metrics.QueryNodeReduceLatency.WithLabelValues(
			fmt.Sprint(t.GetNodeID()),
			metrics.SearchLabel,
			metrics.ReduceSegments).
			Observe(float64(tr.RecordSpan().Milliseconds()))
	}
	defer t.segmentManager.Segment.Unpin(pinnedSegments)

	// 2. reorganize blobs to original search request
	for i := range t.originNqs {
		blob, err := segments.GetSearchResultDataBlob(t.ctx, t.resultBlobs, i)
		if err != nil {
			return err
		}

		var task *StreamingSearchTask
		if i == 0 {
			task = t
		} else {
			task = t.others[i-1]
		}

		// Note: blob is unsafe because get from C
		bs := make([]byte, len(blob))
		copy(bs, blob)

		task.result = &internalpb.SearchResults{
			Base: &commonpb.MsgBase{
				SourceID: t.GetNodeID(),
			},
			Status:         merr.Success(),
			MetricType:     metricType,
			NumQueries:     t.originNqs[i],
			TopK:           t.originTopks[i],
			SlicedBlob:     bs,
			SlicedOffset:   1,
			SlicedNumCount: 1,
			CostAggregation: &internalpb.CostAggregation{
				ServiceTime: tr.ElapseSpan().Milliseconds(),
			},
		}
	}

	return nil
}

func (t *StreamingSearchTask) maybeReturnForEmptyResults(results []*segments.SearchResult,
	metricType string, tr *timerecord.TimeRecorder,
) bool {
	if len(results) == 0 {
		for i := range t.originNqs {
			var task *StreamingSearchTask
			if i == 0 {
				task = t
			} else {
				task = t.others[i-1]
			}

			task.result = &internalpb.SearchResults{
				Base: &commonpb.MsgBase{
					SourceID: t.GetNodeID(),
				},
				Status:         merr.Success(),
				MetricType:     metricType,
				NumQueries:     t.originNqs[i],
				TopK:           t.originTopks[i],
				SlicedOffset:   1,
				SlicedNumCount: 1,
				CostAggregation: &internalpb.CostAggregation{
					ServiceTime: tr.ElapseSpan().Milliseconds(),
				},
			}
		}
		return true
	}
	return false
}

func (t *StreamingSearchTask) streamReduce(ctx context.Context,
	plan *segments.SearchPlan,
	newResult *segments.SearchResult,
	sliceNQs []int64,
	sliceTopKs []int64,
) error {
	if t.streamReducer == nil {
		var err error
		t.streamReducer, err = segments.NewStreamReducer(ctx, plan, sliceNQs, sliceTopKs)
		if err != nil {
			log.Error("Fail to init stream reducer, return")
			return err
		}
	}

	return segments.StreamReduceSearchResult(ctx, newResult, t.streamReducer)
}
