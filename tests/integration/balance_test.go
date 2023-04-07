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
	rowNum        = 3000
	indexType     = "IVF_FLAT"
)

func setUp(t *testing.T) (*MiniCluster, context.Context) {
	ctx := context.Background()
	cluster, err := StartMiniCluster(ctx)
	assert.NoError(t, err)
	err = cluster.Start()
	assert.NoError(t, err)
	return cluster, ctx
}

func TestBalance(t *testing.T) {
	//1. set up cluster
	cluster, ctx := setUp(t)
	defer cluster.Stop()

	//2. prepare data
	collectionName := prefix + funcutil.GenRandomStr()
	schema := constructBasicITSchema(int64Field, floatVecField, dim, collectionName)
	createCollection(t, ctx, cluster, dbName, collectionName, schema, shardNum)
	showCollections(t, cluster, ctx)
	fVecColumn := newFloatVectorFieldData(floatVecField, rowNum, dim)
	hashKeys := generateHashKeys(rowNum)
	insertData(t, cluster, ctx, dbName, collectionName, hashKeys, rowNum, fVecColumn)
	flushAndWaitSegmentFlushed(t, cluster, ctx, dbName, collectionName)

	//3. create index
	createIndex(t, cluster, ctx, collectionName, floatVecField, dim, indexType)
	loadIndex(t, cluster, ctx, dbName, collectionName)

}
