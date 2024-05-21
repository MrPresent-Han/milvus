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
#include <unordered_map>

namespace milvus::segcore{

struct StreamReduceV2ResultPair{


    StreamReduceV2ResultPair(milvus::PkType pk,
                             float distance,
                             std::optional<milvus::GroupByValueType> group_by_value):
            pk_(pk),
            distance_(distance),
            group_by_value_(group_by_value){

    }

    bool
    operator>(const StreamReduceV2ResultPair& other) const{
        if (std::fabs(distance_ - other.distance_) < 0.0000000119) {
            return pk_ < other.pk_;
        }
        return distance_ > other.distance_;
    }

    milvus::PkType pk_;
    float distance_;
    std::optional<milvus::GroupByValueType> group_by_value_;

    void* segment_;
    int64_t seg_off_;
};

//the comparator to heap-sort the new input results
//max-heap as the distances inside the results have been made to have positively-related distance
struct StreamReduceV2SortComparator{
    bool
    operator()(const std::shared_ptr<StreamReduceV2ResultPair> lhs,
            const std::shared_ptr<StreamReduceV2ResultPair> rhs) const{
        return *(rhs.get()) > *(lhs.get());
    }
};

//the comparator to maintain search results maintained in the memory
//min-heap
struct StreamReduceV2GroupComparator{
    bool
    operator()(const std::shared_ptr<StreamReduceV2ResultPair> lhs,
               const std::shared_ptr<StreamReduceV2ResultPair> rhs) const{
        return !(*(rhs.get()) > *(lhs.get()));
    }
};

class SearchVectorGroup{
public:
    SearchVectorGroup(int capacity):capacity_(capacity){

    }

private:
    std::set<std::shared_ptr<StreamReduceV2ResultPair>,
            StreamReduceV2GroupComparator> res_queue_{};

    std::unordered_map<milvus::PkType, std::shared_ptr<StreamReduceV2ResultPair>> pk_map_{};
    int capacity_;

public:
    bool pushPair(std::shared_ptr<StreamReduceV2ResultPair> result_pair);

private:
    bool shouldContinueLessDistance(float distance);
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
        slice_nqs_prefix_sum_.resize(num_slice_ + 1);
        //set up nqs prefixSum
        int64_t sum = 0;
        slice_nqs_prefix_sum_[0] = 0;
        for(int64_t i = 0; i < slice_num; i++) {
            sum += slice_nqs[i];
            slice_nqs_prefix_sum_[i + 1] = sum;
        }

        //set up res_queue, for every qi, set a query_group
        res_group_.resize(sum);
        int64_t slice_idx = 0;
        for(int64_t i = 0; i < sum; i++){
            if(i>=slice_nqs_prefix_sum_[slice_idx+1]) {
                slice_idx++;
            }
            AssertInfo(slice_idx < num_slice_, "slice_idx:{} cannot be larger than num_slice_:{}, must be sth wrong",
                       slice_idx, num_slice_);
            int64_t topK = slice_topKs_[slice_idx];
            res_group_[i] = std::make_shared<SearchVectorGroup>(topK);
        }
    }

private:
    milvus::query::Plan* plan_;
    int64_t* slice_nqs_;
    int64_t* slice_topKs_;
    int64_t num_slice_{0};
    int64_t num_segments_{0};
    std::vector<SearchResult*> search_results_to_merge_{};
    std::vector<std::shared_ptr<SearchVectorGroup>> res_group_{};
    std::vector<int64_t> slice_nqs_prefix_sum_{};

public:
    void
    SetSearchResultsToMerge(std::vector<SearchResult*>& search_results) {
        search_results_to_merge_ = std::move(search_results);
        num_segments_ = search_results_to_merge_.size();
        AssertInfo(num_segments_ > 0, "empty search result");
    }

    void
    MergeReduce();


private:
    void
    StreamReduceOneQuery(int64_t qi, int64_t topK);
};
}