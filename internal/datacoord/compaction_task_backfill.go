package datacoord

import (
	"context"
	"time"

	"go.uber.org/atomic"
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/internal/compaction"
	"github.com/milvus-io/milvus/internal/datacoord/allocator"
	"github.com/milvus-io/milvus/internal/datacoord/session"
	"github.com/milvus-io/milvus/internal/datacoord/task"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/taskcommon"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
)

var _ CompactionTask = (*backfillCompactionTask)(nil)

type backfillCompactionTask struct {
	taskProto atomic.Value // *datapb.CompactionTask
	allocator allocator.Allocator
	meta      CompactionMeta
	handler   Handler
	ievm      IndexEngineVersionManager
}

func newBackfillCompactionTask(t *datapb.CompactionTask, allocator allocator.Allocator, meta CompactionMeta, handler Handler, ievm IndexEngineVersionManager) *backfillCompactionTask {
	task := &backfillCompactionTask{
		allocator: allocator,
		meta:      meta,
		handler:   handler,
		ievm:      ievm,
	}
	task.taskProto.Store(t)
	return task
}

func (t *backfillCompactionTask) GetTaskID() int64 {
	return t.GetTaskProto().GetPlanID()
}

func (t *backfillCompactionTask) GetTaskType() taskcommon.Type {
	return taskcommon.Compaction
}

func (t *backfillCompactionTask) GetTaskState() taskcommon.State {
	return taskcommon.FromCompactionState(t.GetTaskProto().GetState())
}

func (t *backfillCompactionTask) GetTaskProto() *datapb.CompactionTask {
	return t.taskProto.Load().(*datapb.CompactionTask)
}

func (t *backfillCompactionTask) GetTaskSlot() int64 {
	return 0
}

func (t *backfillCompactionTask) SetTaskTime(timeType taskcommon.TimeType, time time.Time) {
	return
}

func (t *backfillCompactionTask) GetTaskTime(timeType taskcommon.TimeType) time.Time {
	return time.Time{}
}

func (t *backfillCompactionTask) GetTaskVersion() int64 {
	return int64(t.GetTaskProto().GetRetryTimes())
}

func (t *backfillCompactionTask) BuildCompactionRequest() (*datapb.CompactionPlan, error) {
	compactionParams, err := compaction.GenerateJSONParams()
	if err != nil {
		return nil, err
	}
	log := log.With(zap.Int64("triggerID", t.GetTaskProto().GetTriggerID()), zap.Int64("PlanID", t.GetTaskProto().GetPlanID()), zap.Int64("collectionID", t.GetTaskProto().GetCollectionID()))
	taskProto := t.taskProto.Load().(*datapb.CompactionTask)
	plan := &datapb.CompactionPlan{
		PlanID:                    taskProto.GetPlanID(),
		StartTime:                 taskProto.GetStartTime(),
		Type:                      taskProto.GetType(),
		Channel:                   taskProto.GetChannel(),
		CollectionTtl:             taskProto.GetCollectionTtl(),
		TotalRows:                 taskProto.GetTotalRows(),
		Schema:                    taskProto.GetSchema(),
		PreAllocatedSegmentIDs:    taskProto.GetPreAllocatedSegmentIDs(),
		SlotUsage:                 t.GetSlotUsage(),
		MaxSize:                   taskProto.GetMaxSize(),
		JsonParams:                compactionParams,
		CurrentScalarIndexVersion: t.ievm.GetCurrentScalarIndexEngineVersion(),
	}
	segIDMap := make(map[int64][]*datapb.FieldBinlog, len(plan.SegmentBinlogs))
	segments := make([]*SegmentInfo, 0, len(taskProto.GetInputSegments()))
	for _, segID := range taskProto.GetInputSegments() {
		segInfo := t.meta.GetHealthySegment(context.TODO(), segID)
		if segInfo == nil {
			return nil, merr.WrapErrSegmentNotFound(segID)
		}
		plan.SegmentBinlogs = append(plan.SegmentBinlogs, &datapb.CompactionSegmentBinlogs{
			SegmentID:           segID,
			CollectionID:        segInfo.GetCollectionID(),
			PartitionID:         segInfo.GetPartitionID(),
			Level:               segInfo.GetLevel(),
			InsertChannel:       segInfo.GetInsertChannel(),
			FieldBinlogs:        segInfo.GetBinlogs(),
			Field2StatslogPaths: segInfo.GetStatslogs(),
			Deltalogs:           segInfo.GetDeltalogs(),
			IsSorted:            segInfo.GetIsSorted(),
			StorageVersion:      segInfo.GetStorageVersion(),
		})
		segIDMap[segID] = segInfo.GetDeltalogs()
		segments = append(segments, segInfo)
	}

	logIDRange, err := PreAllocateBinlogIDs(t.allocator, segments)
	if err != nil {
		return nil, err
	}
	plan.PreAllocatedLogIDs = logIDRange
	plan.BeginLogID = logIDRange.Begin
	WrapPluginContext(taskProto.GetCollectionID(), taskProto.GetSchema().GetProperties(), plan)
	log.Info("Compaction handler refreshed backfill compaction plan", zap.Int64("maxSize", plan.GetMaxSize()),
		zap.Any("PreAllocatedLogIDs", logIDRange), zap.Any("segID2DeltaLogs", segIDMap))
	return plan, nil
}

func (t *backfillCompactionTask) GetSlotUsage() int64 {
	return 0
}

func (t *backfillCompactionTask) GetLabel() string {
	return ""
}

func (t *backfillCompactionTask) SetTask(task *datapb.CompactionTask) {
	t.taskProto.Store(task)
}

func (t *backfillCompactionTask) ShadowClone(opts ...compactionTaskOpt) *datapb.CompactionTask {
	taskProto := t.GetTaskProto()
	if taskProto == nil {
		return nil
	}

	// Create a copy of the task using protobuf Clone
	cloned := proto.Clone(taskProto).(*datapb.CompactionTask)

	// Apply options
	for _, opt := range opts {
		opt(cloned)
	}

	return cloned
}

func (t *backfillCompactionTask) SetNodeID(nodeID int64) error {
	return nil
}

func (t *backfillCompactionTask) NeedReAssignNodeID() bool {
	return false
}

func (t *backfillCompactionTask) SaveTaskMeta() error {
	return nil
}

func (t *backfillCompactionTask) CheckCompactionContainsSegment(segmentID int64) bool {
	return false
}

func (t *backfillCompactionTask) Clean() bool {
	return false
}

func (t *backfillCompactionTask) CreateTaskOnWorker(nodeID int64, cluster session.Cluster) {
	log := log.With(zap.Int64("triggerID", t.GetTaskProto().GetTriggerID()),
		zap.Int64("PlanID", t.GetTaskProto().GetPlanID()),
		zap.Int64("collectionID", t.GetTaskProto().GetCollectionID()),
		zap.Int64("nodeID", nodeID))

	plan, err := t.BuildCompactionRequest()
	if err != nil {
		log.Warn("backfillCompactionTask failed to build compaction request", zap.Error(err))
		err = t.updateAndSaveTaskMeta(setState(datapb.CompactionTaskState_failed), setFailReason(err.Error()))
		if err != nil {
			log.Warn("backfillCompactionTask failed to updateAndSaveTaskMeta", zap.Error(err))
		}
		return
	}

	err = cluster.CreateCompaction(nodeID, plan)
	if err != nil {
		log.Warn("backfillCompactionTask failed to notify compaction tasks to DataNode",
			zap.Int64("planID", t.GetTaskProto().GetPlanID()),
			zap.Int64("nodeID", nodeID),
			zap.Error(err))
		err = t.updateAndSaveTaskMeta(setState(datapb.CompactionTaskState_pipelining), setNodeID(task.NullNodeID))
		if err != nil {
			log.Warn("backfillCompactionTask failed to updateAndSaveTaskMeta", zap.Error(err))
		}
		return
	}

	log.Info("backfillCompactionTask created task on worker", zap.Int64("planID", t.GetTaskProto().GetPlanID()),
		zap.Int64("nodeID", nodeID))

	err = t.updateAndSaveTaskMeta(setState(datapb.CompactionTaskState_executing), setNodeID(nodeID))
	if err != nil {
		log.Warn("backfillCompactionTask failed to updateAndSaveTaskMeta", zap.Error(err))
	}
}

func (t *backfillCompactionTask) QueryTaskOnWorker(cluster session.Cluster) {
	return
}

func (t *backfillCompactionTask) DropTaskOnWorker(cluster session.Cluster) {
	return
}
func (t *backfillCompactionTask) GetPlan() *datapb.CompactionPlan {
	return nil
}

func (t *backfillCompactionTask) GetResult() *datapb.CompactionPlanResult {
	return nil
}

// Process performs the task's state machine
// Note: return True means exit this state machine.
// ONLY return True for Completed, Failed, Timeout
func (t *backfillCompactionTask) Process() bool {
	switch t.GetTaskProto().GetState() {
	case datapb.CompactionTaskState_meta_saved:
		return t.processMetaSaved()
	case datapb.CompactionTaskState_completed:
		return t.processCompleted()
	case datapb.CompactionTaskState_failed:
		return true
	case datapb.CompactionTaskState_timeout:
		return true
	default:
		return false
	}
}

func (t *backfillCompactionTask) processMetaSaved() bool {
	// For backfill compaction, we directly mark it as completed
	// since the actual work is done by updating segment metadata
	err := t.updateAndSaveTaskMeta(setState(datapb.CompactionTaskState_completed))
	if err != nil {
		log.Warn("backfillCompactionTask unable to processMetaSaved",
			zap.Int64("planID", t.GetTaskProto().GetPlanID()),
			zap.Error(err))
		return false
	}
	return t.processCompleted()
}

func (t *backfillCompactionTask) processCompleted() bool {
	log.Info("backfillCompactionTask processCompleted",
		zap.Int64("planID", t.GetTaskProto().GetPlanID()),
		zap.Int64("triggerID", t.GetTaskProto().GetTriggerID()))
	return true
}

func (t *backfillCompactionTask) updateAndSaveTaskMeta(opts ...compactionTaskOpt) error {
	task := t.ShadowClone(opts...)
	t.SetTask(task)
	return t.SaveTaskMeta()
}
