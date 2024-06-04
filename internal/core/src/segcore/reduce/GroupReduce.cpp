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

#include "GroupReduce.h"
#include "log/Log.h"
#include "segcore/SegmentInterface.h"

namespace milvus::segcore{

void
GroupReduceHelper::FilterInvalidSearchResult(SearchResult* search_result){
    //do nothing, for group-by reduce, as we calculate prefix_sum for nq when doing group by and no padding invalid results
    //so there's no need to filter search_result
}

int64_t
GroupReduceHelper::ReduceSearchResultForOneNQ(int64_t qi, int64_t topk, int64_t &offset) {
    std::priority_queue<SearchResultPair*,
            std::vector<SearchResultPair*>,
            SearchResultPairComparator>
            heap;
    pk_set_.clear();
    pairs_.clear();
    pairs_.reserve(num_segments_);
    for (int i = 0; i < num_segments_; i++) {
        auto search_result = search_results_[i];
        auto offset_beg = search_result->topk_per_nq_prefix_sum_[qi];
        auto offset_end = search_result->topk_per_nq_prefix_sum_[qi + 1];
        if (offset_beg == offset_end) {
            continue;
        }
        auto primary_key = search_result->primary_keys_[offset_beg];
        auto distance = search_result->distances_[offset_beg];
        AssertInfo(search_result->group_by_values_.has_value(),
                   "Wrong state, search_result has no group_by_vales for group_by_reduce, must be sth wrong!");
        AssertInfo(search_result->group_by_values_.value().size() == search_result->primary_keys_.size(),
                   "Wrong state, search_result's group_by_values's length is not equal to pks' size!");
        auto group_by_val = search_result->group_by_values_.value()[offset_beg];
        pairs_.emplace_back(
                primary_key,
                distance,
                search_result,
                i,
                offset_beg,
                offset_end,
                std::move(group_by_val));
        heap.push(&pairs_.back());
    }

    // nq has no results for all segments
    if (heap.size() == 0) {
        return 0;
    }

    int64_t group_size = search_results_[0]->group_size_.value();
    int64_t group_by_total_size = group_size * topk;
    int64_t filtered_count = 0;
    auto start = offset;
    std::unordered_map<GroupByValueType, int64_t> group_by_map;

    auto should_filtered = [&](const PkType& pk, const GroupByValueType& group_by_val){
        if(pk_set_.count(pk)!=0) return true;
        if(group_by_map.size()>=topk && group_by_map.count(group_by_val)==0) return true;
        if(group_by_map[group_by_val] > group_size) return true;
        return false;
    };

    while (offset - start < group_by_total_size && !heap.empty()) {
        //fetch value
        auto pilot = heap.top();
        heap.pop();
        auto index = pilot->segment_index_;
        auto pk = pilot->primary_key_;
        AssertInfo(pk != INVALID_PK, "Wrong, search results should have been filtered and invalid_pk should not be existed");
        auto group_by_val = pilot->group_by_value_.value();

        //judge filter
        if (!should_filtered(pk, group_by_val)) {
            pilot->search_result_->result_offsets_.push_back(offset++);
            final_search_records_[index][qi].push_back(pilot->offset_);
            pk_set_.insert(pk);
            group_by_map[group_by_val]+=1;
        } else {
            filtered_count++;
        }

        //move pilot forward
        pilot->advance();
        if (pilot->primary_key_ != INVALID_PK) {
            heap.push(pilot);
        }
    }
    return filtered_count;
}

}