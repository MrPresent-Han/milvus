package httpserver

import (
	"fmt"
	"github.com/milvus-io/milvus/pkg/util/paramtable"
	"testing"

	"github.com/milvus-io/milvus/tests/integration"
	"github.com/stretchr/testify/suite"
)

type HttpServerSuite struct {
	integration.MiniClusterSuite
}

func (s *HttpServerSuite) SetupSuite() {
	port := 20001
	paramtable.Init()
	paramtable.Get().Save(paramtable.Get().HTTPCfg.Port.Key, fmt.Sprintf("%d", port))
	s.MiniClusterSuite.SetupSuite()
}

func (s *HttpServerSuite) TestHttpSearch() {
	
}

func TestHttpSearch(t *testing.T) {
	suite.Run(t, new(HttpServerSuite))
}
