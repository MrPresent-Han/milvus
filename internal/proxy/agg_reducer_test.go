package proxy

import (
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/stretchr/testify/suite"
	"testing"
)

type MilvusAggReduceSuite struct {
	suite.Suite
}

func TestSegCoreAggReduce(t *testing.T) {
	
}

func TestAggReduce(t *testing.T) {
	paramtable.Init()
	suite.Run(t, new(MilvusAggReduceSuite))
}
