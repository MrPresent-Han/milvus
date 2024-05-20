// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include "query/PlanImpl.h"
#include "common/QueryResult.h"
#include <queue>

namespace milvus::segcore{

struct StreamReduceV2Result{
    bool
    operator>(const StreamReduceV2Result& other) const{
        return 0;
    }
};

struct StreamReduceV2ResultComparator{
    bool
    operator()(const std::shared_ptr<StreamReduceV2Result> lhs,
            const std::shared_ptr<StreamReduceV2Result> rhs) const{
        return *(rhs.get()) > *(lhs.get());
    }
};

class StreamReducerHelperV2 {
public:
    explicit StreamReducerHelperV2(milvus::query::Plan* plan,
                                   int64_t* slice_nqs,
                                   int64_t* slice_topKs,
                                   int64_t slice_num):
                                   plan_(plan),
                                   slice_nqs_(slice_nqs),
                                   slice_topKs_(slice_topKs),
                                   num_slice_(slice_num){

    }

private:
    milvus::query::Plan* plan_;
    int64_t* slice_nqs_;
    int64_t* slice_topKs_;
    int64_t num_slice_{0};
    int64_t num_segments_{0};
    std::vector<SearchResult*> search_results_to_merge_;
    std::vector<std::priority_queue<std::shared_ptr<StreamReduceV2Result>,
            std::vector<std::shared_ptr<StreamReduceV2Result>, >;

public:
    void
    SetSearchResultsToMerge(std::vector<SearchResult*>& search_results) {
        search_results_to_merge_ = std::move(search_results);
        num_segments_ = search_results_to_merge_.size();
        AssertInfo(num_segments_ > 0, "empty search result");
    }

    void
    MergeReduce();
};
}