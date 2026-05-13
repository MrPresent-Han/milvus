// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compactor

import (
	"context"
	"fmt"
	sio "io"
	"path"
	"strconv"
	"time"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/cockroachdb/errors"
	"github.com/samber/lo"
	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/internal/compaction"
	"github.com/milvus-io/milvus/internal/metastore/kv/binlog"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/storagecommon"
	"github.com/milvus-io/milvus/internal/storagev2/packed"
	"github.com/milvus-io/milvus/internal/util/function"
	"github.com/milvus-io/milvus/internal/util/hookutil"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/log"
	"github.com/milvus-io/milvus/pkg/v3/proto/datapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexcgopb"
	"github.com/milvus-io/milvus/pkg/v3/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metautil"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// schemaBumpWriter abstracts the packed writer interface for V2/V3 compatibility.
// Both packed.PackedWriter (V2) and packed.FFIPackedWriter (V3) implement WriteRecordBatch,
// but their Close signatures differ, so we use wrappers for both.
// Manifest() returns the V3 manifest path after Close(); V2 always returns "".
type schemaBumpWriter interface {
	WriteRecordBatch(arrow.Record) error
	Close() error
	Manifest() string
}

// ffiWriterWrapper wraps packed.FFIPackedWriter to implement schemaBumpWriter,
// capturing the manifest string returned by Close.
type ffiWriterWrapper struct {
	writer   *packed.FFIPackedWriter
	manifest string
}

func (w *ffiWriterWrapper) WriteRecordBatch(r arrow.Record) error {
	return w.writer.WriteRecordBatch(r)
}

func (w *ffiWriterWrapper) Close() error {
	manifest, err := w.writer.Close()
	if err != nil {
		return err
	}
	w.manifest = manifest
	return nil
}

func (w *ffiWriterWrapper) Manifest() string {
	return w.manifest
}

// v2WriterWrapper wraps packed.PackedWriter to implement schemaBumpWriter.
// Manifest() always returns "" for V2 segments (no manifest concept).
// fileSizes is populated by Close() via CloseAndTell, indexed by column group order.
type v2WriterWrapper struct {
	writer    *packed.PackedWriter
	numGroups int
	fileSizes []int64
}

func (w *v2WriterWrapper) WriteRecordBatch(r arrow.Record) error {
	return w.writer.WriteRecordBatch(r)
}

func (w *v2WriterWrapper) Close() error {
	sizes, err := w.writer.CloseAndTell(w.numGroups)
	if err != nil {
		return err
	}
	w.fileSizes = sizes
	return nil
}

func (w *v2WriterWrapper) Manifest() string {
	return ""
}

type bumpSchemaVersionCompactionTask struct {
	ctx              context.Context
	cancel           context.CancelFunc
	plan             *datapb.CompactionPlan
	compactionParams compaction.Params
	done             chan struct{}
	logIDAlloc       allocator.Interface
	functionRunner   function.FunctionRunner
	chunkManager     storage.ChunkManager
}

func (t *bumpSchemaVersionCompactionTask) Compact() (*datapb.CompactionPlanResult, error) {
	if !funcutil.CheckCtxValid(t.ctx) {
		return nil, t.ctx.Err()
	}
	ctx, span := otel.Tracer(typeutil.DataNodeRole).Start(t.ctx, fmt.Sprintf("BumpSchemaVersionCompact-%d", t.GetPlanID()))
	defer span.End()

	if err := t.preCompact(); err != nil {
		log.Ctx(ctx).Warn("failed to preCompact", zap.Error(err))
		return nil, err
	}
	compactStart := time.Now()
	log := log.Ctx(ctx).With(
		zap.Int64("planID", t.GetPlanID()),
		zap.Int64("collectionID", t.GetCollection()),
	)

	missingFunctions, droppedFieldIDs, existingFields, err := t.schemaBumpDecision()
	if err != nil {
		log.Warn("failed to decide schema bump action", zap.Error(err))
		return nil, err
	}

	log.Info("schema bump compact start",
		zap.Int64("segmentID", t.plan.GetSegmentBinlogs()[0].GetSegmentID()),
		zap.Int32("collectionSchemaVersion", t.plan.GetSchema().GetVersion()),
		zap.Int("missingFunctionCount", len(missingFunctions)),
		zap.Int64s("droppedFieldIDs", droppedFieldIDs),
	)

	var result *datapb.CompactionPlanResult
	if len(missingFunctions) == 0 && len(droppedFieldIDs) == 0 {
		result = t.runSchemaVersionBumpOnly()
	} else if len(droppedFieldIDs) > 0 || len(missingFunctions) != 1 {
		result, err = t.runFullSchemaRewrite(ctx, existingFields)
	} else {
		if err := t.setupFunctionRunner(missingFunctions[0]); err != nil {
			log.Warn("failed to set up function runner", zap.Error(err))
			return nil, err
		}
		defer t.functionRunner.Close()
		result, err = t.runMissingFunctionMaterialization(ctx, t.functionRunner)
	}
	if err != nil {
		log.Warn("schema bump compact failed", zap.Error(err), zap.Duration("compact cost", time.Since(compactStart)))
		return nil, err
	}

	log.Info("schema bump compact done", zap.Duration("compact cost", time.Since(compactStart)))
	return result, nil
}

// preCompact validates the compaction plan and checks context.
func (t *bumpSchemaVersionCompactionTask) preCompact() error {
	if ok := funcutil.CheckCtxValid(t.ctx); !ok {
		return t.ctx.Err()
	}

	// Check segment binlogs: must have exactly one segment
	if len(t.plan.GetSegmentBinlogs()) != 1 {
		return errors.Newf("schema bump compaction plan is illegal, must have exactly one segment, but got %d segments, planID = %d", len(t.plan.GetSegmentBinlogs()), t.GetPlanID())
	}

	segment := t.plan.GetSegmentBinlogs()[0]
	if segment.GetManifest() == "" && len(segment.GetFieldBinlogs()) == 0 {
		return errors.Newf("compaction plan is illegal, segment's field binlogs are empty, planID = %d, segmentID = %d", t.GetPlanID(), segment.GetSegmentID())
	}

	return nil
}

func (t *bumpSchemaVersionCompactionTask) setupFunctionRunner(missingFunction *schemapb.FunctionSchema) error {
	functionRunner, err := function.NewFunctionRunner(t.plan.GetSchema(), missingFunction)
	if err != nil {
		return err
	}
	if functionRunner == nil {
		return errors.New("failed to set up missing function runner")
	}

	if err := t.checkFunctionRunner(functionRunner); err != nil {
		functionRunner.Close()
		return err
	}

	t.functionRunner = functionRunner
	return nil
}

func (t *bumpSchemaVersionCompactionTask) checkFunctionRunner(functionRunner function.FunctionRunner) error {
	switch functionRunner.GetSchema().GetType() {
	case schemapb.FunctionType_BM25:
		functionSchema := functionRunner.GetSchema()

		// 1. Check inputFieldIDs: must have exactly one, and type must be varchar
		inputFieldIDs := functionSchema.GetInputFieldIds()
		if len(inputFieldIDs) != 1 {
			return errors.New("bm25 function should have exactly one input field")
		}
		inputFieldID := inputFieldIDs[0]
		inputField := typeutil.GetField(t.plan.GetSchema(), inputFieldID)
		if inputField == nil {
			return errors.New("input field not found in schema")
		}
		if inputField.GetDataType() != schemapb.DataType_VarChar && inputField.GetDataType() != schemapb.DataType_Text {
			return errors.New("input field data type must be varchar or text for bm25 materialization")
		}

		// 2. Check outputFieldIDs: must have exactly one, and type must be SparseFloatVector
		outputFieldIDs := functionSchema.GetOutputFieldIds()
		if len(outputFieldIDs) != 1 {
			return errors.New("bm25 function should have exactly one output field")
		}
		outputFieldID := outputFieldIDs[0]
		outputField := typeutil.GetField(t.plan.GetSchema(), outputFieldID)
		if outputField == nil {
			return errors.New("output field not found in schema")
		}
		if outputField.GetDataType() != schemapb.DataType_SparseFloatVector {
			return errors.New("output field data type must be sparse float vector for bm25 materialization")
		}

		return nil
	default:
		return errors.New("unsupported function type")
	}
}

func (t *bumpSchemaVersionCompactionTask) runMissingFunctionMaterialization(ctx context.Context, functionRunner function.FunctionRunner) (*datapb.CompactionPlanResult, error) {
	if functionRunner == nil {
		return nil, errors.New("missing function runner is not initialized")
	}
	switch functionRunner.GetSchema().GetType() {
	case schemapb.FunctionType_BM25:
		return t.runBm25Function(ctx, functionRunner)
	default:
		return nil, errors.New("unsupported function type")
	}
}

func (t *bumpSchemaVersionCompactionTask) openBinlogReader(inputFieldID int64) (storage.RecordReader, error) {
	inputField := typeutil.GetField(t.plan.GetSchema(), inputFieldID)
	inputSchema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{inputField}}
	reader, _, err := t.openRecordReader(t.plan.GetSegmentBinlogs()[0], inputSchema)
	return reader, err
}

func (t *bumpSchemaVersionCompactionTask) openRecordReader(segment *datapb.CompactionSegmentBinlogs, schema *schemapb.CollectionSchema) (storage.RecordReader, map[int64]struct{}, error) {
	reader, existingFields, err := newCompactionSegmentRecordReader(t.ctx, segment, schema, t.compactionParams.StorageConfig,
		storage.WithCollectionID(segment.GetCollectionID()),
		storage.WithVersion(segment.GetStorageVersion()),
		storage.WithDownloader(t.chunkManager.MultiRead),
		storage.WithStorageConfig(t.compactionParams.StorageConfig),
	)
	if err != nil {
		log.Ctx(t.ctx).Warn("failed to open compaction segment reader", zap.Error(err))
	}
	return reader, existingFields, err
}

func (t *bumpSchemaVersionCompactionTask) schemaBumpDecision() ([]*schemapb.FunctionSchema, []int64, map[int64]struct{}, error) {
	segment := t.plan.GetSegmentBinlogs()[0]
	existingFields, err := compactionSegmentStorageFields(segment, t.compactionParams.StorageConfig)
	if err != nil {
		return nil, nil, nil, err
	}
	return missingSchemaFunctions(t.plan.GetSchema(), existingFields), droppedSchemaFieldIDs(t.plan.GetSchema(), existingFields), existingFields, nil
}

func (t *bumpSchemaVersionCompactionTask) runSchemaVersionBumpOnly() *datapb.CompactionPlanResult {
	segment := t.plan.GetSegmentBinlogs()[0]
	return &datapb.CompactionPlanResult{
		PlanID: t.plan.GetPlanID(),
		State:  datapb.CompactionTaskState_completed,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           segment.GetSegmentID(),
				NumOfRows:           t.plan.GetTotalRows(),
				InsertLogs:          segment.GetFieldBinlogs(),
				Field2StatslogPaths: segment.GetField2StatslogPaths(),
				Deltalogs:           segment.GetDeltalogs(),
				Channel:             segment.GetInsertChannel(),
				StorageVersion:      segment.GetStorageVersion(),
				Manifest:            segment.GetManifest(),
				ExpirQuantiles:      segment.GetExpirQuantiles(),
			},
		},
		Type: t.plan.GetType(),
	}
}

func (t *bumpSchemaVersionCompactionTask) runFullSchemaRewrite(ctx context.Context, existingFields map[int64]struct{}) (*datapb.CompactionPlanResult, error) {
	segment := t.plan.GetSegmentBinlogs()[0]
	collectionID := segment.GetCollectionID()
	partitionID := segment.GetPartitionID()
	segmentID := segment.GetSegmentID()

	reader, _, err := newCompactionSegmentRecordReaderWithFields(t.ctx, segment, t.plan.GetSchema(), t.compactionParams.StorageConfig, existingFields,
		storage.WithCollectionID(collectionID),
		storage.WithVersion(segment.GetStorageVersion()),
		storage.WithDownloader(t.chunkManager.MultiRead),
		storage.WithStorageConfig(t.compactionParams.StorageConfig),
	)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	materializer, err := NewRecordMaterializer(t.plan.GetSchema(), t.plan.GetSchema().GetFunctions(), existingFields)
	if err != nil {
		return nil, err
	}
	defer materializer.Close()

	writer, err := storage.NewBinlogRecordWriter(ctx,
		collectionID,
		partitionID,
		segmentID,
		t.plan.GetSchema(),
		t.logIDAlloc,
		t.compactionParams.BinLogMaxSize,
		t.plan.GetTotalRows(),
		storage.WithUploader(func(ctx context.Context, kvs map[string][]byte) error {
			return t.chunkManager.MultiWrite(ctx, kvs)
		}),
		storage.WithVersion(t.compactionParams.StorageVersion),
		storage.WithStorageConfig(t.compactionParams.StorageConfig),
		storage.WithUseLoonFFI(t.compactionParams.UseLoonFFI),
	)
	if err != nil {
		return nil, err
	}

	writerClosed := false
	defer func() {
		if !writerClosed {
			writer.Close()
		}
	}()

	var totalRows int64
	for {
		record, err := reader.Next()
		if err != nil {
			if err == sio.EOF {
				break
			}
			return nil, err
		}

		wrapped, err := materializer.Wrap(record)
		if err != nil {
			record.Release()
			return nil, err
		}
		if err := writer.Write(wrapped); err != nil {
			releaseWrappedRecord(wrapped, record)
			return nil, err
		}
		totalRows += int64(wrapped.Len())
		releaseWrappedRecord(wrapped, record)
	}

	if err := writer.Close(); err != nil {
		return nil, err
	}
	writerClosed = true

	binlogs, stats, bm25stats, manifest, expirQuantiles := writer.GetLogs()
	if manifest != "" {
		stats = nil
		bm25stats = nil
	}

	insertLogs := storage.SortFieldBinlogs(binlogs)
	if err := binlog.CompressFieldBinlogs(insertLogs); err != nil {
		return nil, err
	}

	var statsLogs []*datapb.FieldBinlog
	if stats != nil {
		statsLogs = []*datapb.FieldBinlog{stats}
		if err := binlog.CompressFieldBinlogs(statsLogs); err != nil {
			return nil, err
		}
	}

	var bm25StatsLogs []*datapb.FieldBinlog
	if len(bm25stats) > 0 {
		bm25StatsLogs = lo.Values(bm25stats)
		if err := binlog.CompressFieldBinlogs(bm25StatsLogs); err != nil {
			return nil, err
		}
	}

	return &datapb.CompactionPlanResult{
		PlanID: t.plan.GetPlanID(),
		State:  datapb.CompactionTaskState_completed,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           segmentID,
				NumOfRows:           totalRows,
				InsertLogs:          insertLogs,
				Field2StatslogPaths: statsLogs,
				Bm25Logs:            bm25StatsLogs,
				Deltalogs:           segment.GetDeltalogs(),
				Channel:             segment.GetInsertChannel(),
				StorageVersion:      t.compactionParams.StorageVersion,
				Manifest:            manifest,
				ExpirQuantiles:      expirQuantiles,
			},
		},
		Type: t.plan.GetType(),
	}, nil
}

func appendBM25StatsFromArrowArray(stats *storage.BM25Stats, arr arrow.Array) (int, error) {
	binaryArray, ok := arr.(*array.Binary)
	if !ok {
		return 0, errors.Newf("bm25 output field must be arrow binary array, got %T", arr)
	}
	memorySize := 0
	for i := 0; i < binaryArray.Len(); i++ {
		if binaryArray.IsNull(i) {
			continue
		}
		value := binaryArray.Value(i)
		stats.AppendBytes(value)
		memorySize += len(value)
	}
	return memorySize, nil
}

// bumpSchemaVersionWriterResult holds the writer and associated metadata needed after writing.
type bumpSchemaVersionWriterResult struct {
	writer         schemaBumpWriter
	arrowSchema    *arrow.Schema
	outputSchema   *schemapb.CollectionSchema
	columnGroups   []storagecommon.ColumnGroup
	paths          []string // V2 only: log paths for buildMergedLogs
	truePaths      []string // V2 only: true paths for log path resolution
	logIDs         []int64  // V2 only: log IDs corresponding to each column group path, for LogID in FieldBinlog
	fileSizes      []int64  // V2 only: on-disk compressed sizes from CloseAndTell, indexed by column group order
	storageVersion int64
	basePath       string             // V3 only: parsed from existing manifest in setupWriter
	v3Stats        []packed.StatEntry // V3 only: stats to commit atomically with column groups
}

func (t *bumpSchemaVersionCompactionTask) setupWriter(outputField *schemapb.FieldSchema, outputFieldID int64, segment *datapb.CompactionSegmentBinlogs, collectionID, partitionID, segmentID int64) (*bumpSchemaVersionWriterResult, error) {
	outputSchema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{
		outputField,
	}}
	arrowSchema, err := storage.ConvertToArrowSchema(outputSchema, true)
	if err != nil {
		return nil, err
	}
	if segment.GetManifest() == "" && len(segment.GetFieldBinlogs()) == 0 {
		return nil, errors.New("segment field binlogs is empty, wrong state for compaction segments")
	}
	newColumnGroups := []storagecommon.ColumnGroup{
		{
			GroupID: outputFieldID,
			Columns: []int{0},
			Fields:  []int64{outputFieldID},
		},
	}

	var pluginContext *indexcgopb.StoragePluginContext
	if hookutil.IsClusterEncryptionEnabled() {
		ez := hookutil.GetEzByCollProperties(t.plan.GetSchema().GetProperties(), collectionID)
		if ez != nil {
			unsafe := hookutil.GetCipher().GetUnsafeKey(ez.EzID, ez.CollectionID)
			if len(unsafe) > 0 {
				pluginContext = &indexcgopb.StoragePluginContext{
					EncryptionZoneId: ez.EzID,
					CollectionId:     ez.CollectionID,
					EncryptionKey:    string(unsafe),
				}
			}
		}
	}

	// Determine effective storage version: V3 only if segment already has a manifest.
	// V2 segments on V3 clusters must stay V2 — a partial manifest (only new columns)
	// would cause segcore to ignore original binlog data, corrupting the segment.
	storageVersion := t.compactionParams.StorageVersion
	existingManifest := segment.GetManifest()
	if storageVersion == storage.StorageV3 && existingManifest == "" {
		storageVersion = storage.StorageV2 // force V2 path for V2 segment on V3 cluster
	}

	result := &bumpSchemaVersionWriterResult{
		arrowSchema:    arrowSchema,
		outputSchema:   outputSchema,
		columnGroups:   newColumnGroups,
		storageVersion: storageVersion,
	}

	if storageVersion == storage.StorageV3 {
		// V3: extend existing manifest by appending new column groups.
		// Parse basePath and version from the segment's current manifest,
		// so the transaction reads the existing manifest and merges new columns in.
		basePath, existingVersion, err := packed.UnmarshalManifestPath(existingManifest)
		if err != nil {
			return nil, merr.WrapErrServiceInternal("failed to parse existing manifest for V3 schema bump", err.Error())
		}
		ffiWriter, err := packed.NewFFIPackedWriter(basePath, existingVersion, arrowSchema, newColumnGroups, t.compactionParams.StorageConfig, pluginContext)
		if err != nil {
			return nil, err
		}
		// The output field is a new addition to the existing manifest (schema bump).
		// Use AddColumnGroup semantics so Loon does not require the count to match
		// the existing groups in the manifest.
		ffiWriter.AsNewColumnGroups()
		result.writer = &ffiWriterWrapper{writer: ffiWriter}
		result.basePath = basePath
	} else {
		// V2: use PackedWriter with explicit file paths.
		// TODO(schema-bump): no file-size rotation — all rows land in one Parquet file per
		// column group regardless of segment size. Fix by splitting at BinLogMaxSize (64MB)
		// and allocating a new LogID+path per chunk, like MultiSegmentWriter.rotateWriter.
		logIdStart, _, err := t.logIDAlloc.Alloc(uint32(len(newColumnGroups)))
		if err != nil {
			return nil, err
		}
		paths := []string{}
		logIDs := []int64{}
		for _, columnGroup := range newColumnGroups {
			p := metautil.BuildInsertLogPath(t.compactionParams.StorageConfig.GetRootPath(), collectionID, partitionID, segmentID, columnGroup.GroupID, logIdStart)
			paths = append(paths, p)
			logIDs = append(logIDs, logIdStart)
			logIdStart++
		}
		truePaths := lo.Map(paths, func(p string, _ int) string {
			if t.compactionParams.StorageConfig.GetStorageType() == "local" {
				return p
			}
			return path.Join(t.compactionParams.StorageConfig.GetBucketName(), p)
		})
		writer, err := packed.NewPackedWriter(truePaths, arrowSchema, packed.DefaultWriteBufferSize, packed.DefaultMultiPartUploadSize, newColumnGroups, t.compactionParams.StorageConfig, pluginContext)
		if err != nil {
			return nil, err
		}
		result.writer = &v2WriterWrapper{writer: writer, numGroups: len(newColumnGroups)}
		result.paths = paths
		result.truePaths = truePaths
		result.logIDs = logIDs
	}

	return result, nil
}

func (t *bumpSchemaVersionCompactionTask) updateStats(stats *storage.BM25Stats, collectionID, partitionID, segmentID, outputFieldID int64, writerResult *bumpSchemaVersionWriterResult) ([]byte, string, int64, error) {
	cli := t.chunkManager
	statsID, _, err := t.logIDAlloc.Alloc(uint32(1))
	if err != nil {
		return nil, "", 0, err
	}

	bytes, err := stats.Serialize()
	if err != nil {
		return nil, "", 0, err
	}

	if writerResult.storageVersion == storage.StorageV3 {
		// V3: write stats file under manifest basePath/_stats/ directory.
		// Stats will be registered in manifest via AddStatsToManifest after Close().
		// basePath was already parsed from the manifest in setupWriter — reuse it here.
		basePath := writerResult.basePath
		statsRelPath := fmt.Sprintf("_stats/bm25.%d/%d", outputFieldID, statsID)
		absStatsPath := path.Join(basePath, statsRelPath)
		if err := cli.Write(t.ctx, absStatsPath, bytes); err != nil {
			return nil, "", 0, merr.WrapErrServiceInternal("failed to write V3 BM25 stats", err.Error())
		}
		writerResult.v3Stats = append(writerResult.v3Stats, packed.StatEntry{
			Key:   fmt.Sprintf("bm25.%d", outputFieldID),
			Files: []string{absStatsPath}, // C++ converts absolute to relative at commit
			Metadata: map[string]string{
				"memory_size": strconv.FormatInt(int64(len(bytes)), 10),
			},
		})
		return bytes, "", 0, nil // no separate bm25LogPathForResult for V3; statsID unused
	}

	// V2: write stats file to bm25_stats path, return path and ID for result metadata
	statsPath := metautil.JoinIDPath(collectionID, partitionID, segmentID, outputFieldID, statsID)
	bm25LogPath := path.Join(cli.RootPath(), common.SegmentBm25LogPath, statsPath)
	if err := cli.Write(t.ctx, bm25LogPath, bytes); err != nil {
		return nil, "", 0, err
	}

	bm25LogPathForResult := metautil.BuildBm25LogPath(
		t.compactionParams.StorageConfig.GetRootPath(),
		collectionID, partitionID, segmentID, outputFieldID, statsID)
	if t.compactionParams.StorageConfig.GetStorageType() != "local" {
		bm25LogPathForResult = path.Join(t.compactionParams.StorageConfig.GetBucketName(), bm25LogPathForResult)
	}

	return bytes, bm25LogPathForResult, statsID, nil
}

func (t *bumpSchemaVersionCompactionTask) buildMergedLogsV2(segment *datapb.CompactionSegmentBinlogs, writerResult *bumpSchemaVersionWriterResult, sparseFieldMemorySize int, totalRows int64, bytes []byte, bm25LogPathForResult string, bm25LogID int64, outputFieldID int64) ([]*datapb.FieldBinlog, []*datapb.FieldBinlog, error) {
	newInsertLogs := make(map[int64]*datapb.FieldBinlog)
	for i, columnGroup := range writerResult.columnGroups {
		fileSize := writerResult.fileSizes[i]
		logPath := writerResult.paths[i]
		truePath := writerResult.truePaths[i]
		if t.compactionParams.StorageConfig.GetStorageType() != "local" {
			logPath = truePath
		}
		fieldBinlog := &datapb.FieldBinlog{
			FieldID:     columnGroup.GroupID,
			ChildFields: columnGroup.Fields,
			Binlogs: []*datapb.Binlog{
				{
					LogSize:    fileSize,
					MemorySize: int64(sparseFieldMemorySize),
					LogPath:    logPath,
					LogID:      writerResult.logIDs[i],
					EntriesNum: totalRows,
				},
			},
		}
		newInsertLogs[columnGroup.GroupID] = fieldBinlog
	}

	return t.finalizeMergedLogs(segment, newInsertLogs, totalRows, bytes, bm25LogPathForResult, bm25LogID, outputFieldID)
}

func (t *bumpSchemaVersionCompactionTask) buildMergedLogsV3(segment *datapb.CompactionSegmentBinlogs, writerResult *bumpSchemaVersionWriterResult, sparseFieldMemorySize int, totalRows int64, bytes []byte, bm25LogPathForResult string, bm25LogID int64, outputFieldID int64) ([]*datapb.FieldBinlog, []*datapb.FieldBinlog, error) {
	// V3: data is tracked by the manifest. Binlog entries in etcd serve as field-presence
	// markers used by getSegmentBinlogFields (schema bump detection). Each entry must have
	// LogID!=0 and LogPath=="" to pass buildBinlogKvs validation — allocate a real ID from
	// logIDAlloc (same allocator used for V2 LogIDs) so the entry is properly persisted.
	logIDStart, _, err := t.logIDAlloc.Alloc(uint32(len(writerResult.columnGroups)))
	if err != nil {
		return nil, nil, err
	}
	newInsertLogs := make(map[int64]*datapb.FieldBinlog)
	for i, columnGroup := range writerResult.columnGroups {
		fieldBinlog := &datapb.FieldBinlog{
			FieldID:     columnGroup.GroupID,
			ChildFields: columnGroup.Fields,
			Binlogs: []*datapb.Binlog{
				{
					MemorySize: int64(sparseFieldMemorySize),
					EntriesNum: totalRows,
					LogID:      logIDStart + int64(i),
				},
			},
		}
		newInsertLogs[columnGroup.GroupID] = fieldBinlog
	}

	return t.finalizeMergedLogs(segment, newInsertLogs, totalRows, bytes, bm25LogPathForResult, bm25LogID, outputFieldID)
}

func (t *bumpSchemaVersionCompactionTask) finalizeMergedLogs(segment *datapb.CompactionSegmentBinlogs, newInsertLogs map[int64]*datapb.FieldBinlog, totalRows int64, bytes []byte, bm25LogPathForResult string, bm25LogID int64, outputFieldID int64) ([]*datapb.FieldBinlog, []*datapb.FieldBinlog, error) {
	// build new Bm25Logs
	bm25FileSize := int64(len(bytes))
	newBm25Logs := []*datapb.FieldBinlog{
		{
			FieldID: outputFieldID,
			Binlogs: []*datapb.Binlog{
				{
					LogSize:    bm25FileSize,
					MemorySize: bm25FileSize,
					LogPath:    bm25LogPathForResult,
					LogID:      bm25LogID,
					EntriesNum: totalRows,
				},
			},
		},
	}

	// Crash-replay guard: if DC crashed after AlterSegments but before the task-state
	// transition, the output field is already in etcd — skip it to stay idempotent.
	originalInsertLogs := segment.GetFieldBinlogs()
	existingFields := make(map[int64]struct{}, len(originalInsertLogs)*2)
	for _, fb := range originalInsertLogs {
		existingFields[fb.GetFieldID()] = struct{}{}
		for _, childID := range fb.GetChildFields() {
			existingFields[childID] = struct{}{}
		}
	}
	newInsertLogsList := storage.SortFieldBinlogs(newInsertLogs)
	dedupedNew := newInsertLogsList[:0]
	for _, fb := range newInsertLogsList {
		if _, ok := existingFields[fb.GetFieldID()]; ok {
			log.Warn("schema bump crash-replay: output field already in segment binlogs, skipping duplicate",
				zap.Int64("fieldID", fb.GetFieldID()),
				zap.Int64("segmentID", segment.GetSegmentID()))
			continue
		}
		childDup := false
		for _, childID := range fb.GetChildFields() {
			if _, ok := existingFields[childID]; ok {
				childDup = true
				break
			}
		}
		if childDup {
			log.Warn("schema bump crash-replay: output field already in segment binlogs, skipping duplicate",
				zap.Int64("fieldID", fb.GetFieldID()),
				zap.Int64("segmentID", segment.GetSegmentID()))
			continue
		}
		dedupedNew = append(dedupedNew, fb)
	}
	mergedInsertLogs := make([]*datapb.FieldBinlog, 0, len(originalInsertLogs)+len(dedupedNew))
	mergedInsertLogs = append(mergedInsertLogs, originalInsertLogs...)
	mergedInsertLogs = append(mergedInsertLogs, dedupedNew...)

	return mergedInsertLogs, newBm25Logs, nil
}

func (t *bumpSchemaVersionCompactionTask) runBm25Function(ctx context.Context, functionRunner function.FunctionRunner) (*datapb.CompactionPlanResult, error) {
	// 1. set up function schema — duplicate checks from checkFunctionRunner so this path cannot panic
	//    if schema/runner state diverges or call sites change.
	functionSchema := functionRunner.GetSchema()
	inputFieldIDs := functionSchema.GetInputFieldIds()
	if len(inputFieldIDs) != 1 {
		return nil, errors.Newf("bm25 materialization: expected exactly one input field, got %d (planID=%d)", len(inputFieldIDs), t.plan.GetPlanID())
	}
	inputFieldID := inputFieldIDs[0]
	outputFieldIDs := functionSchema.GetOutputFieldIds()
	if len(outputFieldIDs) != 1 {
		return nil, errors.Newf("bm25 materialization: expected exactly one output field, got %d (planID=%d)", len(outputFieldIDs), t.plan.GetPlanID())
	}
	outputFieldID := outputFieldIDs[0]
	outputField := typeutil.GetField(t.plan.GetSchema(), outputFieldID)

	// 2. get segment info
	segment := t.plan.GetSegmentBinlogs()[0]
	collectionID := segment.GetCollectionID()
	partitionID := segment.GetPartitionID()
	segmentID := segment.GetSegmentID()

	log := log.Ctx(ctx).With(
		zap.Int64("planID", t.plan.GetPlanID()),
		zap.Int64("collectionID", collectionID),
		zap.Int64("partitionID", partitionID),
		zap.Int64("segmentID", segmentID),
		zap.Int64("inputFieldID", inputFieldID),
		zap.Int64("outputFieldID", outputFieldID),
	)

	// Track durations for each heavy operation
	var readDuration, computeDuration, writeDuration, updateStatsDuration time.Duration

	// 3. open binlog reader
	reader, err := t.openBinlogReader(inputFieldID)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	// 4. set up writer
	writerResult, err := t.setupWriter(outputField, outputFieldID, segment, collectionID, partitionID, segmentID)
	if err != nil {
		return nil, err
	}

	// Log effective storage version (may differ from cluster version for V2-origin segments).
	log.Info("schema bump writer setup",
		zap.Int64("effectiveStorageVersion", writerResult.storageVersion),
		zap.Int64("clusterStorageVersion", t.compactionParams.StorageVersion),
		zap.Bool("segmentHasManifest", segment.GetManifest() != ""),
		zap.Int("inputFieldBinlogCount", len(segment.GetFieldBinlogs())),
		zap.Strings("v2WritePaths", writerResult.paths),
	)
	writerClosed := false
	defer func() {
		if !writerClosed {
			writerResult.writer.Close()
		}
	}()

	// 5. batch loop: read → compute → write → accumulate stats
	_, span := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "BumpSchemaVersionCompact.batchProcess")
	stats := storage.NewBM25Stats()
	var totalRows int64
	var totalSparseMemorySize int
	materializer, err := NewRecordMaterializerWithRunners(t.plan.GetSchema(), []function.FunctionRunner{functionRunner}, map[int64]struct{}{})
	if err != nil {
		span.End()
		return nil, err
	}
	defer materializer.Close()

	for {
		// read one batch
		readStart := time.Now()
		record, err := reader.Next()
		if err != nil {
			if err == sio.EOF {
				readDuration += time.Since(readStart)
				break
			}
			span.End()
			return nil, err
		}
		readDuration += time.Since(readStart)

		// compute BM25 for this batch through the shared materializer path
		computeStart := time.Now()
		wrapped, err := materializer.Wrap(record)
		computeDuration += time.Since(computeStart)
		if err != nil {
			record.Release()
			span.End()
			return nil, err
		}

		outputCol := wrapped.Column(outputFieldID)
		if outputCol == nil {
			releaseWrappedRecord(wrapped, record)
			span.End()
			return nil, merr.WrapErrServiceInternal(fmt.Sprintf("output field %d not found in materialized record", outputFieldID))
		}
		batchMemSize, err := appendBM25StatsFromArrowArray(stats, outputCol)
		if err != nil {
			releaseWrappedRecord(wrapped, record)
			span.End()
			return nil, err
		}

		// write this batch
		writeStart := time.Now()
		arrowRecord := array.NewRecord(writerResult.arrowSchema, []arrow.Array{outputCol}, int64(wrapped.Len()))
		if err := writerResult.writer.WriteRecordBatch(arrowRecord); err != nil {
			arrowRecord.Release()
			releaseWrappedRecord(wrapped, record)
			span.End()
			return nil, err
		}
		arrowRecord.Release()
		writeDuration += time.Since(writeStart)

		totalRows += int64(wrapped.Len())
		totalSparseMemorySize += batchMemSize
		releaseWrappedRecord(wrapped, record)
	}
	span.End()

	// 6. write stats file (must happen before Close for V3 — stats are committed atomically)
	_, span2 := otel.Tracer(typeutil.DataNodeRole).Start(ctx, "BumpSchemaVersionCompact.updateStats")
	startTime := time.Now()
	bytes, bm25LogPathForResult, bm25LogID, err := t.updateStats(stats, collectionID, partitionID, segmentID, outputFieldID, writerResult)
	updateStatsDuration = time.Since(startTime)
	span2.End()
	if err != nil {
		return nil, err
	}

	log.Info("schema bump bm25 stats written",
		zap.String("bm25LogPath", bm25LogPathForResult),
		zap.Int64("bm25LogID", bm25LogID),
		zap.Int("bm25StatsBytes", len(bytes)),
		zap.Int64("storageVersion", writerResult.storageVersion),
		zap.Int("v3StatsCount", len(writerResult.v3Stats)),
	)

	// Close writer — for V3, this commits new column groups to the existing manifest
	// (version N → N+1). BM25 stats are added separately via AddStatsToManifest.
	if err := writerResult.writer.Close(); err != nil {
		return nil, err
	}
	writerClosed = true
	// For V2: capture on-disk compressed file sizes written by CloseAndTell.
	if v2w, ok := writerResult.writer.(*v2WriterWrapper); ok {
		writerResult.fileSizes = v2w.fileSizes
		log.Info("schema bump V2 insert binlogs written",
			zap.Strings("logPaths", writerResult.paths),
			zap.Int64s("logIDs", writerResult.logIDs),
			zap.Int64s("fileSizeBytes", v2w.fileSizes),
			zap.Int64("totalRows", totalRows),
		)
	}
	// Read the manifest produced by Close(); "" for V2, real path for V3.
	manifestPath := writerResult.writer.Manifest()
	if manifestPath != "" {
		log.Info("schema bump V3 writer closed, manifest produced", zap.String("manifestPath", manifestPath))
	}

	// For V3: register BM25 stats in manifest (version N+1 → N+2).
	// Channel-level scheduler exclusion guarantees no concurrent manifest modification.
	if writerResult.storageVersion == storage.StorageV3 && len(writerResult.v3Stats) > 0 {
		newManifest, err := packed.AddStatsToManifest(
			manifestPath, t.compactionParams.StorageConfig, writerResult.v3Stats)
		if err != nil {
			return nil, merr.WrapErrServiceInternal("failed to add BM25 stats to V3 manifest", err.Error())
		}
		log.Info("schema bump V3 bm25 stats added to manifest",
			zap.String("oldManifest", manifestPath),
			zap.String("newManifest", newManifest),
		)
		manifestPath = newManifest
	}

	// 7. build merged logs
	var mergedInsertLogs, mergedBm25Logs []*datapb.FieldBinlog
	if writerResult.storageVersion == storage.StorageV3 {
		mergedInsertLogs, mergedBm25Logs, err = t.buildMergedLogsV3(segment, writerResult, totalSparseMemorySize, totalRows, bytes, bm25LogPathForResult, bm25LogID, outputFieldID)
	} else {
		mergedInsertLogs, mergedBm25Logs, err = t.buildMergedLogsV2(segment, writerResult, totalSparseMemorySize, totalRows, bytes, bm25LogPathForResult, bm25LogID, outputFieldID)
	}
	if err != nil {
		return nil, err
	}

	// 8. manifest path is already set from Close() and optionally updated by AddStatsToManifest above.

	// 9. return compaction result
	// For V3: BM25 stats are embedded in the manifest (committed atomically with
	// column groups), so Bm25Logs should be nil — PackSegmentLoadInfo skips
	// Bm25Logs when ManifestPath is set, relying on StatsResolver to read from manifest.
	resultBm25Logs := mergedBm25Logs
	if writerResult.storageVersion == storage.StorageV3 {
		resultBm25Logs = nil
	}

	ret := &datapb.CompactionPlanResult{
		PlanID: t.plan.GetPlanID(),
		State:  datapb.CompactionTaskState_completed,
		Segments: []*datapb.CompactionSegment{
			{
				SegmentID:           segmentID,
				NumOfRows:           totalRows,
				InsertLogs:          mergedInsertLogs,
				Bm25Logs:            resultBm25Logs,
				Field2StatslogPaths: segment.GetField2StatslogPaths(),
				Deltalogs:           segment.GetDeltalogs(),
				Channel:             segment.GetInsertChannel(),
				StorageVersion:      writerResult.storageVersion,
				Manifest:            manifestPath,
			},
		},
		Type: t.plan.GetType(),
	}
	log.Info("schema bump compaction completed",
		zap.Int64("numOfRows", totalRows),
		zap.Int("mergedInsertLogsCount", len(mergedInsertLogs)),
		zap.Int("mergedBm25LogsCount", len(mergedBm25Logs)),
		zap.Int("resultBm25LogsCount", len(resultBm25Logs)),
		zap.String("manifestPath", manifestPath),
		zap.Int64("effectiveStorageVersion", writerResult.storageVersion),
		zap.Int32("collectionSchemaVersion", t.plan.GetSchema().GetVersion()),
		zap.Duration("readDuration", readDuration),
		zap.Duration("computeDuration", computeDuration),
		zap.Duration("writeDuration", writeDuration),
		zap.Duration("updateStatsDuration", updateStatsDuration),
	)
	return ret, nil
}

func (t *bumpSchemaVersionCompactionTask) Complete() {
	if t.done != nil {
		select {
		case t.done <- struct{}{}:
		default:
		}
	}
}

func (t *bumpSchemaVersionCompactionTask) Stop() {
	if t.cancel != nil {
		t.cancel()
	}
	if t.done != nil {
		<-t.done
	}
}

func (t *bumpSchemaVersionCompactionTask) GetPlanID() typeutil.UniqueID {
	return t.plan.GetPlanID()
}

func (t *bumpSchemaVersionCompactionTask) GetCollection() typeutil.UniqueID {
	// Get collection ID from the first segment binlog
	if len(t.plan.GetSegmentBinlogs()) > 0 {
		return t.plan.GetSegmentBinlogs()[0].GetCollectionID()
	}
	return 0
}

func (t *bumpSchemaVersionCompactionTask) GetChannelName() string {
	return t.plan.GetChannel()
}

func (t *bumpSchemaVersionCompactionTask) GetCompactionType() datapb.CompactionType {
	return t.plan.GetType()
}

func (t *bumpSchemaVersionCompactionTask) GetSlotUsage() int64 {
	return t.plan.GetSlotUsage()
}

func (t *bumpSchemaVersionCompactionTask) GetStorageConfig() *indexpb.StorageConfig {
	return t.compactionParams.StorageConfig
}

var _ Compactor = (*bumpSchemaVersionCompactionTask)(nil)

func NewBumpSchemaVersionCompactionTask(ctx context.Context, cm storage.ChunkManager, plan *datapb.CompactionPlan, compactionParams compaction.Params) *bumpSchemaVersionCompactionTask {
	ctx, cancel := context.WithCancel(ctx)
	return &bumpSchemaVersionCompactionTask{
		ctx:              ctx,
		cancel:           cancel,
		plan:             plan,
		compactionParams: compactionParams,
		done:             make(chan struct{}, 1),
		logIDAlloc:       allocator.NewLocalAllocator(plan.GetPreAllocatedLogIDs().GetBegin(), plan.GetPreAllocatedLogIDs().GetEnd()),
		chunkManager:     cm,
	}
}
