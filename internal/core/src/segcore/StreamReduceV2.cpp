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

#include "StreamReduceV2.h"
namespace milvus::segcore{


void StreamReducerHelperV2::MergeReduce() {
    for(int64_t slice_idx = 0; slice_idx < num_slice_; slice_idx++){
        auto nq_begin = slice_nqs_prefix_sum_[slice_idx];
        auto nq_end = slice_nqs_prefix_sum_[slice_idx + 1];
        for (int64_t qi = nq_begin; qi < nq_end; qi++) {
            StreamReduceOneQuery(qi, slice_topKs_[slice_idx]);
        }
    }
}

void
StreamReducerHelperV2::StreamReduceOneQuery(int64_t qi, int64_t topK) {
    //1. build up heap for new input search results
    std::priority_queue<std::shared_ptr<StreamReduceV2ResultPair>,
            std::vector<std::shared_ptr<StreamReduceV2ResultPair>>,
            StreamReduceV2SortComparator> heap;

    for(int i = 0; i < num_segments_; i++){
        auto search_result = search_results_to_merge_[i];
        auto offset_beg = search_result->topk_per_nq_prefix_sum_[qi];
        auto offset_end = search_result->topk_per_nq_prefix_sum_[qi+1];
        if(offset_beg == offset_end){
            continue;
        }
        auto primary_key = search_result->primary_keys_[offset_beg];
        auto distance = search_result->distances_[offset_beg];
        if (search_result->group_by_values_.has_value()) {
            AssertInfo(
                    search_result->group_by_values_.value().size() > offset_beg,
                    "Wrong size for group_by_values size to "
                    "ReduceSearchResultForOneNQ:{}, not enough for"
                    "required offset_beg:{}",
                    search_result->group_by_values_.value().size(),
                    offset_beg);
        }
        auto result_pair = std::make_shared<StreamReduceV2ResultPair>(
                primary_key,
                distance,
                search_result->group_by_values_.has_value() &&
                search_result->group_by_values_.value().size() > offset_beg
                ? std::make_optional(
                        search_result->group_by_values_.value().at(offset_beg))
                : std::nullopt
                );
        heap.push(result_pair);
    }
    if(heap.empty()){
        return;
    }

    //2. pop and sort
    auto search_group = res_group_[qi];
    //res_group must have been initialized when constructing the StreamReducer, impossible to out-of-bound here
    while(!heap.empty()){
        auto pilot = heap.top();
        heap.pop();
        auto resultPair = std::make_shared<StreamReduceV2ResultPair>(pilot->pk_, pilot->distance_, pilot->group_by_value_);
        if(!search_group->pushPair(resultPair)){
            //as the results in the new heap are all out of range
            //no need to traverse the rest of results
            break;
        }
    }

}

bool SearchVectorGroup::shouldContinueLessDistance(float distance) {
    if(res_queue_.empty()) return true;
    auto min_pair_it = res_queue_.cbegin();
    auto min_pair = min_pair_it->get();
    return (res_queue_.size() == capacity_) && (min_pair->distance_ >= distance);
}

bool SearchVectorGroup::pushPair(std::shared_ptr<StreamReduceV2ResultPair> new_pair) {
    bool should_continueForLess = shouldContinueLessDistance(new_pair->distance_);
    if(!should_continueForLess){
        return false;
    }

    auto new_pk = new_pair->pk_;
    auto pk_it = pk_map_.find(new_pk);
    //1. pk has existed
    if(pk_it!=pk_map_.end()){
        auto existed_pair = pk_it->second;
        if(existed_pair->distance_ >= new_pair->distance_){
            //new pair is no better than existed one, just ignore and make outer ops continue
            return should_continueForLess;
        }
        //new_pair is better, remove the existed pair
        res_queue_.erase(existed_pair);
        pk_map_.erase(pk_it);
        res_queue_.insert(new_pair);
        pk_map_[new_pk] = new_pair;
        return should_continueForLess;
    }

    //2. pk not existed for now
    res_queue_.insert(new_pair);
    pk_map_[new_pk] = new_pair;
    if(res_queue_.size() <= capacity_){
        return shouldContinueLessDistance(new_pair->distance_);
    } else {
        auto min_pair_it = res_queue_.cbegin();
        auto min_pk = min_pair_it->get()->pk_;
        pk_map_.erase(min_pk);
        res_queue_.erase(min_pair_it);
        return should_continueForLess;
    }
}

}