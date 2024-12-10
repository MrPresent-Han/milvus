package segments

import (
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"github.com/stretchr/testify/suite"
	"testing"
)

type AggReduceSuite struct {
	suite.Suite
}

func TestAggReduce1(t *testing.T) {

}

func TestAggReduce(t *testing.T) {
	paramtable.Init()
	suite.Run(t, new(AggReduceSuite))
}
