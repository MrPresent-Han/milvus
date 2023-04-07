package integration

import (
	"context"
	"github.com/milvus-io/milvus/pkg/util/funcutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	prefix        = "TestBalance"
	dbName        = ""
	int64Field    = "int64"
	floatVecField = "fvec"
	dim           = 128
	shardNum      = int32(2)
	rowNum        = 1000
	indexType     = "FLAT"
)

func setUp(t *testing.T, dataNodeCount int, queryNodeCount int) (*MiniCluster, context.Context) {
	ctx := context.Background()
	setCountOption := func(cluster *MiniCluster) {
		cluster.clusterConfig.DataNodeNum = dataNodeCount
		cluster.clusterConfig.QueryNodeNum = queryNodeCount
	}
	cluster, err := StartMiniCluster(ctx, setCountOption)
	assert.NoError(t, err)
	err = cluster.Start()
	assert.NoError(t, err)
	return cluster, ctx
}

func TestBalance(t *testing.T) {
	//1. set up cluster
	cluster, ctx := setUp(t, 1, 4)
	defer cluster.Stop()

	//2. create collection
	collectionName := prefix + funcutil.GenRandomStr()
	schema := constructBasicITSchema(int64Field, floatVecField, dim, collectionName)
	createCollection(t, ctx, cluster, dbName, collectionName, schema, shardNum)
	showCollections(t, cluster, ctx)

	//3. insert data in 20 rounds, generating 40 segments
	roundCount := 20
	for i := 0; i < roundCount; i++ {
		fVecColumn := newFloatVectorFieldData(floatVecField, rowNum, dim)
		insertData(t, cluster, ctx, dbName, collectionName, rowNum, fVecColumn)
		flushAndWaitSegmentFlushed(t, cluster, ctx, dbName, collectionName)
	}

	//3. create index
	createIndex(t, cluster, ctx, collectionName, floatVecField, dim, indexType)

	//4. load segments async onto 4 query nodes previously launched
	loadIndex(t, cluster, ctx, dbName, collectionName, true)

	//5. start four more query node
	resizeConfig := ClusterConfig{
		QueryNodeNum: 8,
	}
	cluster.UpdateClusterSize(resizeConfig)

	//
	cluster.queryNodes

}
