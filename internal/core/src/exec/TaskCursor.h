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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "exec/Driver.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include "plan/PlanNode.h"

namespace milvus {
namespace exec {

struct CursorParameters {
    std::shared_ptr<plan::PlanNode> plannode_;

    int32_t destination_ = 0;
    // Maximum number of drivers per pipeline.
    int32_t max_drivers_ = 1;
    // Maximum number of split groups processed concurrently.
    int32_t max_num_concurrent_splitgroup_ = 1;

    std::shared_ptr<QueryContext> query_context_;

    ExecutionStrategy execution_strategy_{};

    // Number of splits groups the task will be processing. Must be 1 for
    // ungrouped execution.
    int num_splitgroups_{1};
};

class TaskQueue {
 public:
    struct TaskQueueEntry {
        RowVectorPtr vector_;
    };

    void
    SetNumProducers(int32_t n) {
        num_producers_ = n;
    }

    BlockingReason
    Enqueue(RowVectorPtr vector, ContinueFuture* future);

    RowVectorPtr
    Dequeue();

    void
    Close();

    bool
    HasNext();

 private:
    std::queue<TaskQueueEntry> queue_;
    std::optional<int32_t> num_producers_;
    int32_t producers_finished_{0};
    std::mutex mutex_;
    // std::vector<ContinuePromise> producer_unblock_promises_;
    bool consumer_blocked_ = false;
    ContinuePromise consumer_promise_;
    ContinueFuture consumer_future_;
    bool closed_ = false;
};

class TaskCursor {
 public:
    explicit TaskCursor(const CursorParameters& params);

    ~TaskCursor() {
        queue_->close();
        if (task_ && !at_end_) {
            task->RequestCancel();
        }
    }

    void
    Start();

    bool
    MoveNext();

    bool
    HasNext();

    RowVectorPtr&
    Current() {
        return currnet_;
    }

    const std::shared_ptr<Task>&
    task() {
        return task_;
    }

 private:
    static std::atomic<int32_t> serial_;
    const int32_t max_drivers_;
    const int32_t num_concurrent_splitgroups_;
    const int32_t num_splitgroups_;

    std::shared_ptr<folly::Executor> executor_;
    bool started_{false};
    std::shared_ptr<TaskQueue> queue_;
    std::shared_ptr<exec::Task> task_;
    RowVectorPtr current_;
    bool at_end_{false};
};

class RowCursor {
 public:
    explicit RowCursor(CursorParameter& params) {
        cursor_ = std::make_unique<TaskCursor>(params);
    }

    bool
    Next();

    bool
    HasNext();

    std::shared_ptr<Task>
    task() const {
        return cursor_->task();
    }

 private:
    std::unique_ptr<TaskCursor> cursor_;
    int64_t current_row_{0};
    int64_t num_rows_{0};
};
}  // namespace exec
}  // namespace milvus