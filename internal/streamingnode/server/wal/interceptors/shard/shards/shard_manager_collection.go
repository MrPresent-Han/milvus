package shards

import (
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/streamingnode/server/wal/interceptors/shard/policy"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/streamingpb"
	"github.com/milvus-io/milvus/pkg/v2/streaming/util/message"
)

// CheckIfCollectionCanBeCreated checks if a collection can be created.
// It returns false if the collection cannot be created.
func (m *shardManagerImpl) CheckIfCollectionCanBeCreated(collectionID int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.checkIfCollectionCanBeCreated(collectionID)
}

// checkIfCollectionCanBeCreated checks if a collection can be created.
func (m *shardManagerImpl) checkIfCollectionCanBeCreated(collectionID int64) error {
	if _, ok := m.collections[collectionID]; ok {
		return ErrCollectionExists
	}
	return nil
}

// CheckIfCollectionExists checks if a collection can be dropped.
func (m *shardManagerImpl) CheckIfCollectionExists(collectionID int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.checkIfCollectionExists(collectionID)
}

// checkIfCollectionExists checks if a collection exists.
func (m *shardManagerImpl) checkIfCollectionExists(collectionID int64) error {
	if _, ok := m.collections[collectionID]; !ok {
		return ErrCollectionNotFound
	}
	return nil
}

// CreateCollection creates a new partition manager when create collection message is written into wal.
// After CreateCollection is called, the ddl and dml on the collection can be applied.
func (m *shardManagerImpl) CreateCollection(msg message.ImmutableCreateCollectionMessageV1) {
	collectionID := msg.Header().CollectionId
	partitionIDs := msg.Header().PartitionIds
	vchannel := msg.VChannel()
	timetick := msg.TimeTick()
	logger := m.Logger().With(log.FieldMessage(msg))

	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.checkIfCollectionCanBeCreated(collectionID); err != nil {
		logger.Warn("collection already exists")
		return
	}

	m.collections[collectionID] = newCollectionInfo(vchannel, partitionIDs)
	for partitionID := range m.collections[collectionID].PartitionIDs {
		uniqueKey := PartitionUniqueKey{CollectionID: collectionID, PartitionID: partitionID}
		if _, ok := m.partitionManagers[uniqueKey]; ok {
			logger.Warn("partition already exists", zap.Int64("partitionID", partitionID))
			continue
		}
		m.partitionManagers[uniqueKey] = newPartitionSegmentManager(
			m.ctx,
			m.Logger(),
			m.wal,
			m.pchannel,
			vchannel,
			collectionID,
			partitionID,
			make(map[int64]*segmentAllocManager),
			m.txnManager,
			timetick,
			m.metrics,
		)
	}
	logger.Info("collection created in segment assignment service", zap.Int64s("partitionIDs", partitionIDs),
		zap.Uint64("timetick", timetick))

	// schema := msg.MustBody().GetCollectionSchema()
	// schemaOfVChannel := &streamingpb.CollectionSchemaOfVChannel{
	// 	Schema:             schema,
	// 	CheckpointTimeTick: timetick,
	// 	State:              streamingpb.VChannelSchemaState_VCHANNEL_SCHEMA_STATE_NORMAL,
	// }
	// m.collections[collectionID].Schemas = append(m.collections[collectionID].Schemas, schemaOfVChannel)
	// log.Info("hc====append new collection schema", zap.Any("schemaOfVChannel", schemaOfVChannel),
	// 	zap.Uint64("schemaVersion", schema.GetSchemaVersion()))
	m.updateMetrics()
}

// DropCollection drops the collection and all the partitions and segments belong to it when drop collection message is written into wal.
// After DropCollection is called, no more segments can be assigned to the collection.
// Any dml and ddl for the collection will be rejected.
func (m *shardManagerImpl) DropCollection(msg message.ImmutableDropCollectionMessageV1) {
	collectionID := msg.Header().CollectionId
	logger := m.Logger().With(log.FieldMessage(msg))

	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.checkIfCollectionExists(collectionID); err != nil {
		logger.Warn("collection not exists")
		return
	}

	collectionInfo := m.collections[collectionID]
	delete(m.collections, collectionID)
	// remove all partition and segment
	partitionIDs := make([]int64, 0, len(collectionInfo.PartitionIDs))
	segmentIDs := make([]int64, 0, len(collectionInfo.PartitionIDs))
	for partitionID := range collectionInfo.PartitionIDs {
		uniqueKey := PartitionUniqueKey{CollectionID: collectionID, PartitionID: partitionID}
		pm, ok := m.partitionManagers[uniqueKey]
		if !ok {
			logger.Warn("partition not exists", zap.Int64("partitionID", partitionID))
			continue
		}
		// Flush all segments and fence assign to the partition manager.
		segments := pm.FlushAndDropPartition(policy.PolicyCollectionRemoved())
		partitionIDs = append(partitionIDs, partitionID)
		segmentIDs = append(segmentIDs, segments...)
		delete(m.partitionManagers, uniqueKey)
	}
	logger.Info("collection removed", zap.Int64s("partitionIDs", partitionIDs), zap.Int64s("segmentIDs", segmentIDs))
	m.updateMetrics()
}

func (m *shardManagerImpl) AppendNewCollectionSchema(msg message.ImmutableSchemaChangeMessageV2) {
	log.Info("hc====AppendNewSchemaChange", zap.Any("msg", msg))
	header := msg.Header()
	collectionID := header.CollectionId
	schema := msg.MustBody().Schema
	timetick := msg.TimeTick()

	m.mu.Lock()
	defer m.mu.Unlock()

	nweCollectionSchema := &streamingpb.CollectionSchemaOfVChannel{
		Schema:             schema,
		CheckpointTimeTick: timetick,
		State:              streamingpb.VChannelSchemaState_VCHANNEL_SCHEMA_STATE_NORMAL,
	}
	m.collections[collectionID].Schemas = append(m.collections[collectionID].Schemas, nweCollectionSchema)
	log.Info("hc====AppendedNewCollectionSchema", zap.Any("msg", msg))
}

func (m *shardManagerImpl) AppendNewCollectionSchemaFromCreateCollection(msg message.ImmutableCreateCollectionMessageV1) {
	log.Info("hc====AppendNewCreateCollection", zap.Any("msg", msg))
	header := msg.Header()
	collectionID := header.CollectionId
	schema := msg.MustBody().GetCollectionSchema()
	timetick := msg.TimeTick()
	schema.SchemaVersion = timetick

	m.mu.Lock()
	defer m.mu.Unlock()

	newCollectionSchema := &streamingpb.CollectionSchemaOfVChannel{
		Schema:             schema,
		CheckpointTimeTick: timetick,
		State:              streamingpb.VChannelSchemaState_VCHANNEL_SCHEMA_STATE_NORMAL,
	}
	m.collections[collectionID].Schemas = append(m.collections[collectionID].Schemas, newCollectionSchema)
	log.Info("hc====AppendedNewCollectionSchema", zap.Any("msg", msg), zap.Uint64("schemaVersion", schema.GetSchemaVersion()),
		zap.Uint64("timetick", timetick))
}

func (m *shardManagerImpl) CheckIfCollectionSchemaVersionMatch(collectionID int64, schemaVersion uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.checkIfCollectionSchemaVersionMatch(collectionID, schemaVersion)
}

func (m *shardManagerImpl) checkIfCollectionSchemaVersionMatch(collectionID int64, schemaVersion uint64) error {
	if _, ok := m.collections[collectionID]; !ok {
		log.Warn("hc====collection not found", zap.Int64("collectionID", collectionID))
		return ErrCollectionNotFound
	}
	if len(m.collections[collectionID].Schemas) == 0 {
		log.Warn("hc====collection schema not found", zap.Int64("collectionID", collectionID))
		//hc----here need to handle the case that the collection schema is not found in the recovery storage
		return nil
	}
	collectionSchemaVersion := m.collections[collectionID].Schemas[len(m.collections[collectionID].Schemas)-1].GetSchema().GetSchemaVersion()
	if collectionSchemaVersion != schemaVersion {
		log.Warn("hc====collection schema version not match", zap.Int64("collectionID", collectionID), zap.Uint64("schemaVersion", schemaVersion),
			zap.Any("collectionSchemaVersion", collectionSchemaVersion))
		return ErrCollectionSchemaVersionNotMatch
	}
	log.Info("hc====collection schema version match", zap.Int64("collectionID", collectionID), zap.Uint64("schemaVersion", schemaVersion), zap.Any("collectionSchemaVersion", collectionSchemaVersion))
	return nil
}
