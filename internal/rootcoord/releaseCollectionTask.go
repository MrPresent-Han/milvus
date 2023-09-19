package rootcoord

import (
	"context"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/pkg/util/typeutil"
)

type releaseCollectionTask struct {
	baseTask
	Req *milvuspb.ReleaseCollectionRequest
}

func (t *releaseCollectionTask) Prepare(ctx context.Context) error {
	collectionMeta, err := t.core.meta.GetCollectionByName(ctx, t.Req.GetDbName(),
		t.Req.GetCollectionName(), typeutil.MaxTimestamp)
	if err != nil {
		return err
	}

	redoTask := newBaseRedoTask(t.core.stepExecutor)
	//release channels and segments on querynodes
	redoTask.AddAsyncStep(&releaseCollectionOnQueryNodesStep{
		baseStep:     baseStep{core: t.core},
		collectionID: collectionMeta.CollectionID,
	})
	//unwatch dml channels on datanodes
	redoTask.AddAsyncStep(&unwatchChannelsStep{
		baseStep:     baseStep{core: t.core},
		collectionID: collectionMeta.CollectionID,
		channels: collectionChannels{
			virtualChannels:  collectionMeta.VirtualChannelNames,
			physicalChannels: collectionMeta.PhysicalChannelNames,
		},
	})
	return redoTask.Execute(ctx)
}
