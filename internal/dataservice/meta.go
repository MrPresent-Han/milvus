package dataservice

import (
	"fmt"
	"strconv"
	"sync"

	log "github.com/sirupsen/logrus"

	"github.com/zilliztech/milvus-distributed/internal/proto/datapb"
	"github.com/zilliztech/milvus-distributed/internal/proto/schemapb"

	"github.com/zilliztech/milvus-distributed/internal/util/typeutil"

	"github.com/golang/protobuf/proto"
	"github.com/zilliztech/milvus-distributed/internal/errors"
	"github.com/zilliztech/milvus-distributed/internal/kv"
)

type (
	UniqueID       = typeutil.UniqueID
	Timestamp      = typeutil.Timestamp
	collectionInfo struct {
		ID     UniqueID
		Schema *schemapb.CollectionSchema
	}
	meta struct {
		client      kv.TxnBase                       // client of a reliable kv service, i.e. etcd client
		collID2Info map[UniqueID]*collectionInfo     // collection id to collection info
		segID2Info  map[UniqueID]*datapb.SegmentInfo // segment id to segment info

		allocator allocator
		ddLock    sync.RWMutex
	}
)

func newMetaTable(kv kv.TxnBase, allocator allocator) (*meta, error) {
	mt := &meta{
		client:      kv,
		collID2Info: make(map[UniqueID]*collectionInfo),
		segID2Info:  make(map[UniqueID]*datapb.SegmentInfo),
		allocator:   allocator,
	}
	err := mt.reloadFromKV()
	if err != nil {
		return nil, err
	}
	return mt, nil
}

func (meta *meta) reloadFromKV() error {
	_, values, err := meta.client.LoadWithPrefix("segment")
	if err != nil {
		return err
	}

	for _, value := range values {
		segmentInfo := &datapb.SegmentInfo{}
		err = proto.UnmarshalText(value, segmentInfo)
		if err != nil {
			return err
		}
		meta.segID2Info[segmentInfo.SegmentID] = segmentInfo
	}

	return nil
}

func (meta *meta) AddCollection(collectionInfo *collectionInfo) error {
	meta.ddLock.Lock()
	defer meta.ddLock.Unlock()
	if _, ok := meta.collID2Info[collectionInfo.ID]; ok {
		return fmt.Errorf("collection %s with id %d already exist", collectionInfo.Schema.Name, collectionInfo.ID)
	}
	meta.collID2Info[collectionInfo.ID] = collectionInfo
	return nil
}

func (meta *meta) DropCollection(collID UniqueID) error {
	meta.ddLock.Lock()
	defer meta.ddLock.Unlock()

	if _, ok := meta.collID2Info[collID]; !ok {
		return errors.Errorf("can't find collection. id = " + strconv.FormatInt(collID, 10))
	}
	delete(meta.collID2Info, collID)
	for id, segment := range meta.segID2Info {
		if segment.CollectionID != collID {
			continue
		}
		delete(meta.segID2Info, id)
		if err := meta.removeSegmentInfo(id); err != nil {
			log.Printf("remove segment info failed, %s", err.Error())
			_ = meta.reloadFromKV()
		}
	}
	return nil
}

func (meta *meta) HasCollection(collID UniqueID) bool {
	meta.ddLock.RLock()
	defer meta.ddLock.RUnlock()
	_, ok := meta.collID2Info[collID]
	return ok
}

func (meta *meta) BuildSegment(collectionID UniqueID, partitionID UniqueID, channelRange []string) (*datapb.SegmentInfo, error) {
	id, err := meta.allocator.allocID()
	if err != nil {
		return nil, err
	}
	ts, err := meta.allocator.allocTimestamp()
	if err != nil {
		return nil, err
	}

	return &datapb.SegmentInfo{
		SegmentID:      id,
		CollectionID:   collectionID,
		PartitionID:    partitionID,
		InsertChannels: channelRange,
		OpenTime:       ts,
		CloseTime:      0,
		NumRows:        0,
		MemSize:        0,
	}, nil
}

func (meta *meta) AddSegment(segmentInfo *datapb.SegmentInfo) error {
	meta.ddLock.Lock()
	defer meta.ddLock.Unlock()
	if _, ok := meta.segID2Info[segmentInfo.SegmentID]; !ok {
		return fmt.Errorf("segment %d already exist", segmentInfo.SegmentID)
	}
	meta.segID2Info[segmentInfo.SegmentID] = segmentInfo
	if err := meta.saveSegmentInfo(segmentInfo); err != nil {
		_ = meta.reloadFromKV()
		return err
	}
	return nil
}

func (meta *meta) UpdateSegment(segmentInfo *datapb.SegmentInfo) error {
	meta.ddLock.Lock()
	defer meta.ddLock.Unlock()

	meta.segID2Info[segmentInfo.SegmentID] = segmentInfo
	if err := meta.saveSegmentInfo(segmentInfo); err != nil {
		_ = meta.reloadFromKV()
		return err
	}
	return nil
}

func (meta *meta) GetSegment(segID UniqueID) (*datapb.SegmentInfo, error) {
	meta.ddLock.RLock()
	defer meta.ddLock.RUnlock()

	segmentInfo, ok := meta.segID2Info[segID]
	if !ok {
		return nil, errors.Errorf("GetSegmentByID:can't find segment id = %d", segID)
	}
	return segmentInfo, nil
}

func (meta *meta) CloseSegment(segID UniqueID, closeTs Timestamp) error {
	meta.ddLock.Lock()
	defer meta.ddLock.Unlock()

	segInfo, ok := meta.segID2Info[segID]
	if !ok {
		return errors.Errorf("DropSegment:can't find segment id = " + strconv.FormatInt(segID, 10))
	}

	segInfo.CloseTime = closeTs

	err := meta.saveSegmentInfo(segInfo)
	if err != nil {
		_ = meta.reloadFromKV()
		return err
	}
	return nil
}

func (meta *meta) GetCollection(collectionID UniqueID) (*collectionInfo, error) {
	meta.ddLock.RLock()
	defer meta.ddLock.RUnlock()

	collectionInfo, ok := meta.collID2Info[collectionID]
	if !ok {
		return nil, fmt.Errorf("collection %d not found", collectionID)
	}
	return collectionInfo, nil
}

func (meta *meta) saveSegmentInfo(segmentInfo *datapb.SegmentInfo) error {
	segBytes := proto.MarshalTextString(segmentInfo)

	return meta.client.Save("/segment/"+strconv.FormatInt(segmentInfo.SegmentID, 10), segBytes)
}

func (meta *meta) removeSegmentInfo(segID UniqueID) error {
	return meta.client.Remove("/segment/" + strconv.FormatInt(segID, 10))
}
