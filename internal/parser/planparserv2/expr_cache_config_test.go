package planparserv2

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

func TestExprCacheMaxEntriesHotRefresh(t *testing.T) {
	params := paramtable.Get()
	config := &params.ProxyCfg.ExprCacheMaxEntries
	original := config.GetValue()
	configureExprCache()
	exprCache.Purge()
	t.Cleanup(func() {
		require.NoError(t, params.Save(config.Key, original))
		exprCache.Purge()
	})

	require.NoError(t, params.Save(config.Key, "3"))
	for i := 0; i < 4; i++ {
		_, err := handleInternal(fmt.Sprintf("Int64Field == %d", i))
		require.NoError(t, err)
	}
	require.Equal(t, 3, exprCache.Len())
	_, oldestExists := exprCache.Get("Int64Field == 0")
	assert.False(t, oldestExists)

	require.NoError(t, params.Save(config.Key, "1"))
	require.Equal(t, 1, exprCache.Len())
	_, newestExists := exprCache.Get("Int64Field == 3")
	assert.True(t, newestExists)

	require.NoError(t, params.Save(config.Key, "2"))
	_, err := handleInternal("Int64Field == 4")
	require.NoError(t, err)
	assert.Equal(t, 2, exprCache.Len())
}
