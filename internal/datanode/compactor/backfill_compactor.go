package compactor

import (
	"context"

	"github.com/cockroachdb/errors"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/internal/compaction"
	"github.com/milvus-io/milvus/internal/flushcommon/io"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/function"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
	"go.uber.org/zap"
)

type backfillCompactionTask struct {
	ctx              context.Context
	cancel           context.CancelFunc
	binlogIO         io.BinlogIO
	plan             *datapb.CompactionPlan
	compactionParams compaction.Params
	done             chan struct{}
}

func (t *backfillCompactionTask) Compact() (*datapb.CompactionPlanResult, error) {
	// For backfill compaction, we update segment metadata to reflect schema changes
	// This is a metadata-only operation that doesn't require actual data processing

	// Create result segments with updated schema versions
	// resultSegments := make([]*datapb.CompactionSegment, 0, len(t.plan.GetSegmentBinlogs()))

	// for _, segmentBinlog := range t.plan.GetSegmentBinlogs() {
	// 	// For backfill compaction, we create a "new" segment that's essentially the same
	// 	// as the input segment but with updated schema version metadata
	// 	resultSegment := &datapb.CompactionSegment{
	// 		SegmentID:      segmentBinlog.GetSegmentID(),
	// 		Channel:        t.plan.GetChannel(),
	// 		NumOfRows:      0, // This will be filled by the DataCoord when processing the result
	// 		IsSorted:       segmentBinlog.GetIsSorted(),
	// 		StorageVersion: segmentBinlog.GetStorageVersion(),
	// 		// The key point: this segment now has the current schema version
	// 		// The actual schema version update will be handled by DataCoord
	// 	}
	// 	resultSegments = append(resultSegments, resultSegment)
	// }

	// return &datapb.CompactionPlanResult{
	// 	PlanID:   t.plan.GetPlanID(),
	// 	State:    datapb.CompactionTaskState_completed,
	// 	Segments: resultSegments,
	// 	Channel:  t.plan.GetChannel(),
	// }, nil
	log.Warn("hc====sn===Start to run backfill function")
	backfillFunctions := t.plan.GetFunctions()
	if len(backfillFunctions) != 1 {
		return nil, errors.New("backfill functions should be exactly one")
	}
	backfillFunction := backfillFunctions[0]
	functionRunner, err := function.NewFunctionRunner(t.plan.GetSchema(), backfillFunction)
	if err != nil {
		return nil, err
	}
	if functionRunner == nil {
		return nil, errors.New("backfill function runner is nil")
	}
	err = t.runBackfillFunction(functionRunner)
	if err != nil {
		return nil, err
	}
	log.Warn("hc====sn===Finish to run backfill function")
	return nil, nil
}

func (t *backfillCompactionTask) runBackfillFunction(functionRunner function.FunctionRunner) error {
	switch functionRunner.GetSchema().GetType() {
	case schemapb.FunctionType_BM25:
		return t.runBm25Function(functionRunner)
	default:
		return errors.New("unsupported function type")
	}
}

func (t *backfillCompactionTask) runBm25Function(functionRunner function.FunctionRunner) error {
	//1. set up function schema
	functionSchema := functionRunner.GetSchema()
	inputFieldIDs := functionSchema.GetInputFieldIds()
	if len(inputFieldIDs) != 1 {
		return errors.New("bm25 function should have exactly one input field")
	}
	inputFieldID := inputFieldIDs[0]
	inputField := typeutil.GetField(t.plan.GetSchema(), inputFieldID)
	if inputField == nil {
		return errors.New("input field not found")
	}

	//2. get input data
	segment := t.plan.GetSegmentBinlogs()[0]
	collectionID := segment.GetCollectionID()
	partitionID := segment.GetPartitionID()
	segmentID := segment.GetSegmentID()
	if err := binlog.DecompressBinLogWithRootPath(t.compactionParams.StorageConfig.GetRootPath(),
		storage.InsertBinlog, collectionID, partitionID,
		segmentID, segment.GetFieldBinlogs()); err != nil {
		log.Ctx(t.ctx).Warn("Decompress insert binlog error", zap.Error(err))
		return err
	}

	//3. run function
	// output, err := functionRunner.BatchRun(inputData)
	// if err != nil {
	// 	return err
	// }

	// functionRunner.BatchRun()
	return nil
}

func (t *backfillCompactionTask) Complete() {
	if t.done != nil {
		select {
		case t.done <- struct{}{}:
		default:
		}
	}
}

func (t *backfillCompactionTask) Stop() {
	if t.cancel != nil {
		t.cancel()
	}
	if t.done != nil {
		<-t.done
	}
}

func (t *backfillCompactionTask) GetPlanID() typeutil.UniqueID {
	return t.plan.GetPlanID()
}

func (t *backfillCompactionTask) GetCollection() typeutil.UniqueID {
	// Get collection ID from the first segment binlog
	if len(t.plan.GetSegmentBinlogs()) > 0 {
		return t.plan.GetSegmentBinlogs()[0].GetCollectionID()
	}
	return 0
}

func (t *backfillCompactionTask) GetChannelName() string {
	return t.plan.GetChannel()
}

func (t *backfillCompactionTask) GetCompactionType() datapb.CompactionType {
	return t.plan.GetType()
}

func (t *backfillCompactionTask) GetSlotUsage() int64 {
	return t.plan.GetSlotUsage()
}

var _ Compactor = (*backfillCompactionTask)(nil)

func NewBackfillCompactionTask(ctx context.Context, binlogIO io.BinlogIO, plan *datapb.CompactionPlan, compactionParams compaction.Params) *backfillCompactionTask {
	ctx, cancel := context.WithCancel(ctx)
	return &backfillCompactionTask{
		ctx:              ctx,
		cancel:           cancel,
		binlogIO:         binlogIO,
		plan:             plan,
		compactionParams: compactionParams,
		done:             make(chan struct{}, 1),
	}
}
