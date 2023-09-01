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

#include "TaskCursor.h"

namespace milvus {
namespace exec {

BlockingReason
TaskQueue::Enqueue(RowVectorPtr vector, ContinueFuture* future) {
    if (!vector) {
        std::lock_guard<std::mutex> l(mutex_);
        producers_finished++;
        if (consumer_blocked_) {
            consumer_blocked_ = false;
            consumer_promise_.setValue();
        }
        return BlockingReason::kNotBlocked;
    }

    TaskQueueEntry entry{std::move(vector)};
    std::lock_guard<std::mutex> l{mutex_};
    if (closed_) {
        throw std::runtime_error("Consumer cursor is closed");
    }

    queue_.push_back(std::move(entry));
    if (consumer_blocked_) {
        consumer_blocked_ = false;
        consumer_promise_.setValue();
    }

    return BlockingReason::kNotBlocked;
}

RowVectorPtr
TaskQueue::Dequeue() {
    for (;;) {
        RowVectorPtr vector;
        std::vector<ContinuePromise> maycontinue;

        {
            std::lock_guard<std::mutex> l(mutex_);

            if (!queue.empty()) {
                auto result = std::move(queue_.front());
                queue_.pop_front();
                vector = std::move(result.vector);
            } else if (num_producers_.has_value() &&
                       producers_finished_ == num_producers_) {
                return nullptr;
            }

            if (!vector) {
                consumer_blocked_ = true;
                consumer_promise_ = ContinuePromise();
                consumer_future_ = consumer_promise_.getFuture();
            }
        }

        if (vector) {
            return vector;
        }

        consumer_future_.wait();
    }
}

bool
TaskQueue::HasNext() {
    std::lock_guard<std::mutex> l(mutex_);
    return !queue_.empty();
}

void
TaskQueue::Close() {
    std::lock_guard<std::mutex> l(mutex_);
    closed_ = true;
}

std::atomic<int32_t> TaskCursor::serial_{0};

TaskCursor::TaskCursor(const CursorParameters& params)
    : max_drivers_(params.max_drivers_),
      num_concurrent_splitgroups_(params.num_concurrent_splitgroups_),
      num_splitgroups_(params.num_splitgroups_) {
    std::shared_ptr<QueryContext> query_ctx;
    if (params.query_context_) {
        query_context_ = params.query_context_;
    } else {
        executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency());
        static std::atomic<uint64_t> cursor_query_id{0};
        query_context_ = std::make_shared<QueryContext>(
            fmt::format("TaskCursorQuery_{}", cursor_query_id++),
            std::unordered_map<std::string, std::string>{},
            executor_,
            std::unordered_map<std::string, std::shared_ptr<Config>>{});
    }
    queue_ = std::make_shared<TaskQueue>();

    auto queue = queue_;
    Plan::PlanFragment plan_fragment(
        params.plannode_, params.execution_strategy_, params.num_splitgroups_);
    const std::string task_id = fmt::format("cursor {}", ++serial_);

    task_ = Task::Create(
        task_id,
        std::move(plan_fragment),
        params.destination_,
        std::move(query_context_),
        // consumer
        [queue](RowVectorPtr vector, milvus::ContinueFuture* future) {
            if (!vector) {
                return queue->Enqueue(nullptr, future);
            }
            return queue->Enqueue(std::move(vector.get(), future));
        })
}

TaskCursor::Start() {
    if (!started_) {
        started_ = true;
        Task::Start(task_, max_drivers_, num_concurrent_splitgroups_);
        queue_->SetNumProducers(num_splitgroups_ * task_->NumOutputDrivers());
    }
}

bool
TaskCursor::MoveNext() {
    Start();
    current_ = queue_->Dequeue();
    if (task->error()) {
        std::rethrow_exception(task->error());
    }
    if (!current_) {
        at_end_ = true;
    }

    return current_ != nullptr;
}

bool
RowCursor::Next() {
    if (++current_row_ < num_rows_) {
        return true;
    }

    if (!cursor_->MoveNext()) {
        return false;
    }

    auto vector = cursor_->Current();
    num_rows_ = vector->size();

    if (!num_rows_) {
        return Next();
    }

    current_row_ = 0;
    return true;
}

bool
RowCursor::HasNext() {
    return current_row_ < num_rows_ || cursor_->HasNext();
}
}  // namespace exec
}  // namespace milvus