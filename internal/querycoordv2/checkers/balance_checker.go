// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package checkers

import (
	"context"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/querycoordv2/meta"
	"github.com/samber/lo"
	"time"

	"github.com/milvus-io/milvus/internal/querycoordv2/balance"
	. "github.com/milvus-io/milvus/internal/querycoordv2/params"
	"github.com/milvus-io/milvus/internal/querycoordv2/task"
)

// BalanceChecker checks the cluster distribution and generates balance tasks.
type BalanceChecker struct {
	baseChecker
	balance.Balance
	meta      *meta.Meta
	scheduler task.Scheduler
}

func NewBalanceChecker(meta *meta.Meta, balancer balance.Balance, scheduler task.Scheduler) *BalanceChecker {
	return &BalanceChecker{
		Balance:   balancer,
		meta:      meta,
		scheduler: scheduler,
	}
}

func (b *BalanceChecker) Description() string {
	return "BalanceChecker checks the cluster distribution and generates balance tasks"
}

func (b *BalanceChecker) shouldDoBalance() (bool, []int64) {
	//if configured no auto balance, skip balance
	if !Params.QueryCoordCfg.AutoBalance.GetAsBool() {
		return false, nil
	}
	//if there are still tasks running in the scheduler, skip balance
	//to avoid increase instability
	if b.scheduler.GetSegmentTaskNum() != 0 || b.scheduler.GetChannelTaskNum() != 0 {
		return false, nil
	}
	// loading collection should skip balance
	ids := b.meta.CollectionManager.GetAll()
	loadedCollections := lo.Filter(ids, func(cid int64, _ int) bool {
		return b.meta.CalculateLoadStatus(cid) == querypb.LoadStatus_Loaded
	})
	return len(loadedCollections) == 0, loadedCollections
}

func (b *BalanceChecker) Check(ctx context.Context) []task.Task {
	ret := make([]task.Task, 0)
	shouldDoBalance, collectionsToBalance := b.shouldDoBalance()
	if !shouldDoBalance {
		return ret
	}

	segmentPlans, channelPlans := b.Balance.Balance()

	tasks := balance.CreateSegmentTasksFromPlans(ctx, b.ID(), Params.QueryCoordCfg.SegmentTaskTimeout.GetAsDuration(time.Millisecond), segmentPlans)
	task.SetPriorityWithFunc(func(t task.Task) task.Priority {
		if t.Priority() == task.TaskPriorityHigh {
			return task.TaskPriorityHigh
		}
		return task.TaskPriorityLow
	}, tasks...)
	ret = append(ret, tasks...)

	tasks = balance.CreateChannelTasksFromPlans(ctx, b.ID(), Params.QueryCoordCfg.ChannelTaskTimeout.GetAsDuration(time.Millisecond), channelPlans)
	ret = append(ret, tasks...)
	return ret
}
