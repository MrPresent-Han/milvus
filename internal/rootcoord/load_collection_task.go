package rootcoord

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type loadCollectionTask struct {
	baseTask
	Req          *milvuspb.LoadCollectionRequest
	dbID         UniqueID
	collectionID UniqueID
}

func (t *loadCollectionTask) Prepare(ctx context.Context) error {
	//hc---the meaning of MaxTimestamp here?
	db, err := t.core.meta.GetDatabaseByName(ctx, t.Req.GetDbName(), typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	t.dbID = db.ID

	//hc---collection exist?
	targetCollection, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetDbName(), t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		return err
	}
	t.collectionID = targetCollection.CollectionID

	return nil
}

func (t *loadCollectionTask) Execute(ctx context.Context) error {
	undoTask := newBaseUndoTask(t.core.stepExecutor)
	collectionMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.DbName,
		t.Req.CollectionName, typeutil.MaxTimestamp)
	if err != nil {
		return err
	}

	//load on queryNodes
	undoTask.AddStep(&nullStep{}, &releaseCollectionOnQueryNodesStep{
		baseStep:     baseStep{core: t.core},
		collectionID: t.collectionID,
	})
	undoTask.AddStep(&loadCollectionOnQueryNodes{
		baseStep:     baseStep{core: t.core},
		collectionID: t.collectionID,
	}, &nullStep{})

	//watch channels on dataNodes
	undoTask.AddStep(&nullStep{}, &unwatchChannelsStep{
		baseStep:     baseStep{core: t.core},
		collectionID: t.collectionID,
		channels: collectionChannels{
			virtualChannels:  collectionMeta.VirtualChannelNames,
			physicalChannels: collectionMeta.PhysicalChannelNames,
		},
	})
	undoTask.AddStep(&watchChannelsStep{
		baseStep: baseStep{core: t.core},
	}, &nullStep{})

	return undoTask.Execute(ctx)
}
