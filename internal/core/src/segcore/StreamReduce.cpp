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

#include "StreamReduce.h"
#include "SegmentInterface.h"
#include "segcore/Utils.h"

namespace milvus::segcore {

   /* void
    MergeReduceHelper::AssembleMergedResult() {
        auto merge_search_result = std::make_unique<SearchResult>();
        std::vector<PkType> pks;
        std::vector<float> distances;
        std::vector<GroupByValueType> group_by_values;

        for(int i = 0; i < num_slices_; i++) {
            //1. calculate the result count for current nq
            auto nq_begin = slice_nqs_prefix_sum_[i];
            auto nq_end = slice_nqs_prefix_sum_[i+1];
            int64_t nq_result_count = 0;
            for (auto search_result: search_results_){
                AssertInfo(search_result->topk_per_nq_prefix_sum_.size() ==
                           search_result->total_nq_ + 1,
                           "incorrect topk_per_nq_prefix_sum_ size in search result");
                nq_result_count += (search_result->topk_per_nq_prefix_sum_[nq_end] -
                                    search_result->topk_per_nq_prefix_sum_[nq_begin]);
            }

            for(auto qi = nq_begin; qi < nq_end; qi++){
                for(auto search_result : search_results_) {
                    AssertInfo(search_result != nullptr,
                               "null search result when assembling merged result");
                    if (search_result->result_offsets_.size() == 0) continue;
                    auto topK_start = search_result->topk_per_nq_prefix_sum_[qi];
                    auto topK_end = search_result->topk_per_nq_prefix_sum_[qi+1];
                    for(auto k = topK_start; k < topK_end; k++){
                        auto loc = search_result->result_offsets_[k];
                        AssertInfo(loc < nq_result_count && loc >= 0,
                                   "invalid loc when GetSearchResultDataSlice, loc = " +
                                   std::to_string(loc) + ", result_count = " +
                                   std::to_string(nq_result_count));
                        //auto offset = final_search_records_[]
                        //append pk, offset, distance, mergeOutputFields
                    }
                }
            }

        }
    }
   */

    void
    MergeReduceHelper::FillEntryData() {
        for (auto search_result: search_results_to_merge_) {
            auto segment = static_cast<milvus::segcore::SegmentInterface*>(
                    search_result->segment_);
            segment->FillTargetEntry(plan_, *search_result);
        }
    }

    void
    MergeReduceHelper::AssembleMergedResult() {
        std::unique_ptr<MergedSearchResult> new_merged_result = std::make_unique<MergedSearchResult>();
        std::vector<PkType> new_merged_pks;
        std::vector<float> new_merged_distances;
        std::vector<GroupByValueType> new_merged_groupBy_vals;
        bool need_handle_groupBy = plan_->plan_node_->search_info_.group_by_field_id_.has_value();
        int valid_size = 0;
        if(search_results_to_merge_.size() > 0) {
            for(int i = 0; i < num_slice_; i++) {
                auto nq_begin = slice_nqs_prefix_sum_[i];
                auto nq_end = slice_nqs_prefix_sum_[i+1];
                int64_t result_count = 0;
                for (auto search_result : search_results_to_merge_) {
                    AssertInfo(search_result->topk_per_nq_prefix_sum_.size() ==
                               search_result->total_nq_ + 1,
                               "incorrect topk_per_nq_prefix_sum_ size in search result");
                    result_count += search_result->topk_per_nq_prefix_sum_[nq_end] -
                                    search_result->topk_per_nq_prefix_sum_[nq_begin];
                }
                result_count += merged_search_result->topk_per_nq_prefix_sum_[nq_end] - merged_search_result->topk_per_nq_prefix_sum_[nq_begin];
                std::vector<MergeBase> new_result_pairs(result_count);
                int nq_base_offset = valid_size;
                valid_size+=result_count;
                new_merged_pks.resize(valid_size);
                new_merged_distances.resize(valid_size);
                if(need_handle_groupBy){
                    new_merged_groupBy_vals.resize(valid_size);
                }
                for (auto qi = nq_begin; qi < nq_end; qi++){
                      for (auto search_result: search_results_to_merge_){
                          AssertInfo(search_result != nullptr,
                                     "null search result when reorganize");
                          if (search_result->result_offsets_.size() == 0) {
                              continue;
                          }
                          auto topK_start = search_result->topk_per_nq_prefix_sum_[qi];
                          auto topK_end = search_result->topk_per_nq_prefix_sum_[qi + 1];
                          for(auto ki = topK_start; ki < topK_end; ki++){
                              auto loc = search_result->result_offsets_[ki];
                              AssertInfo(loc < result_count && loc >= 0,
                                         "invalid loc when GetSearchResultDataSlice, loc = " +
                                         std::to_string(loc) + ", result_count = " +
                                         std::to_string(result_count));

                              new_merged_pks[nq_base_offset + loc] = search_result->primary_keys_[ki];
                              new_merged_distances[nq_base_offset + loc] = search_result->distances_[ki];
                              if(need_handle_groupBy){
                                  new_merged_groupBy_vals[nq_base_offset + loc] = search_result->group_by_values_.value()[ki];
                              }
                              new_result_pairs[loc] = {&search_result->output_fields_data_, ki};
                          }
                      }
                      auto topK_start = merged_search_result->topk_per_nq_prefix_sum_[qi];
                      auto topK_end = merged_search_result->topk_per_nq_prefix_sum_[qi + 1];
                      for(auto ki = topK_start; ki < topK_end; ki++){
                         auto loc = merged_search_result->result_offsets_[ki];
                         AssertInfo(loc < result_count && loc >= 0,
                                   "invalid loc when GetSearchResultDataSlice, loc = " +
                                   std::to_string(loc) + ", result_count = " +
                                   std::to_string(result_count));

                         new_merged_pks[nq_base_offset + loc] = merged_search_result->primary_keys_[ki];
                         new_merged_distances[nq_base_offset + loc] = merged_search_result->distances_[ki];
                         if(need_handle_groupBy){
                            new_merged_groupBy_vals[nq_base_offset + loc] = merged_search_result->group_by_values_.value()[ki];
                         }
                         new_result_pairs[loc] = {&merged_search_result->output_fields_data_, ki};
                      }
                }
            }
        }
    }

    void
    MergeReduceHelper::MergeReduce() {
        FillPrimaryKey();
        ReduceResultData();
        RefreshSearchResult();
        FillEntryData();
        AssembleMergedResult();
    }


    void
    MergeReduceHelper::ReduceResultData() {
        for (int i = 0; i < num_segments_; i++) {
            auto search_result = search_results_to_merge_[i];
            auto result_count = search_result->get_total_result_count();
            AssertInfo(search_result != nullptr,
                       "search result must not equal to nullptr");
            AssertInfo(search_result->distances_.size() == result_count,
                       "incorrect search result distance size");
            AssertInfo(search_result->seg_offsets_.size() == result_count,
                       "incorrect search result seg offset size");
            AssertInfo(search_result->primary_keys_.size() == result_count,
                       "incorrect search result primary key size");
        }
        std::unique_ptr<SearchResult> search_result = std::make_unique<SearchResult>();
        for (int64_t slice_index = 0; slice_index < slice_nqs_.size(); slice_index++) {
            auto nq_begin = slice_nqs_prefix_sum_[slice_index];
            auto nq_end = slice_nqs_prefix_sum_[slice_index + 1];

            int64_t offset = 0;
            for (int64_t qi = nq_begin; qi < nq_end; qi++) {
                StreamReduceSearchResultForOneNQ(qi, slice_topKs_[slice_index], offset);
            }
        }
    }

    void
    MergeReduceHelper::FillPrimaryKey() {
        uint32_t valid_index = 0;
        for (auto& search_result : search_results_to_merge_) {
            // skip when results num is 0
            if (search_result->unity_topK_ == 0) {
                continue;
            }
            FilterInvalidSearchResult(search_result);
            auto segment = static_cast<SegmentInterface*>(search_result->segment_);
            if (search_result->get_total_result_count() > 0) {
                segment->FillPrimaryKeys(plan_, *search_result);
                search_results_to_merge_[valid_index++] = search_result;
            }
        }
        search_results_to_merge_.resize(valid_index);
        num_segments_ = search_results_to_merge_.size();
    }

    void
    MergeReduceHelper::FilterInvalidSearchResult(SearchResult* search_result) {
        auto nq = search_result->total_nq_;
        auto topK = search_result->unity_topK_;
        AssertInfo(search_result->seg_offsets_.size() == nq * topK,
                   "wrong seg offsets size, size = " +
                   std::to_string(search_result->seg_offsets_.size()) +
                   ", expected size = " + std::to_string(nq * topK));
        AssertInfo(search_result->distances_.size() == nq * topK,
                   "wrong distances size, size = " +
                   std::to_string(search_result->distances_.size()) +
                   ", expected size = " + std::to_string(nq * topK));
        std::vector<int64_t> real_topks(nq, 0);
        uint32_t valid_index = 0;
        auto segment = static_cast<SegmentInterface*>(search_result->segment_);
        auto& offsets = search_result->seg_offsets_;
        auto& distances = search_result->distances_;
        if (search_result->group_by_values_.has_value()) {
            AssertInfo(search_result->distances_.size() ==
                       search_result->group_by_values_.value().size(),
                       "wrong group_by_values size, size:{}, expected size:{} ",
                       search_result->group_by_values_.value().size(),
                       search_result->distances_.size());
        }

        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < topK; ++j) {
                auto index = i * topK + j;
                if (offsets[index] != INVALID_SEG_OFFSET) {
                    AssertInfo(0 <= offsets[index] &&
                               offsets[index] < segment->get_row_count(),
                               fmt::format("invalid offset {}, segment {} with "
                                           "rows num {}, data or index corruption",
                                           offsets[index],
                                           segment->get_segment_id(),
                                           segment->get_row_count()));
                    real_topks[i]++;
                    offsets[valid_index] = offsets[index];
                    distances[valid_index] = distances[index];
                    if (search_result->group_by_values_.has_value())
                        search_result->group_by_values_.value()[valid_index] =
                                search_result->group_by_values_.value()[index];
                    valid_index++;
                }
            }
        }
        offsets.resize(valid_index);
        distances.resize(valid_index);
        if (search_result->group_by_values_.has_value())
            search_result->group_by_values_.value().resize(valid_index);

        search_result->topk_per_nq_prefix_sum_.resize(nq + 1);
        std::partial_sum(real_topks.begin(),
                         real_topks.end(),
                         search_result->topk_per_nq_prefix_sum_.begin() + 1);
    }

    void
    MergeReduceHelper::StreamReduceSearchResultForOneNQ(int64_t qi,
                                                        int64_t topK,
                                                        int64_t& offset) {
        //1. clear heap for preceding left elements
        while (!heap_.empty()) {
            heap_.pop();
        }
        pk_set_.clear();
        group_by_val_set_.clear();

        //2. push new search results into sort-heap
        for (int i = 0; i < num_segments_; i++) {
            auto search_result = search_results_to_merge_[i];
            auto offset_beg = search_result->topk_per_nq_prefix_sum_[qi];
            auto offset_end = search_result->topk_per_nq_prefix_sum_[qi + 1];
            if (offset_beg == offset_end) {
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

            auto result_pair = std::make_shared<SearchResultPair>(
                    primary_key,
                    distance,
                    search_result,
                    i,
                    offset_beg,
                    offset_end,
                    search_result->group_by_values_.has_value() &&
                    search_result->group_by_values_.value().size() > offset_beg
                    ? std::make_optional(
                            search_result->group_by_values_.value().at(offset_beg))
                    : std::nullopt);
            heap_.push(result_pair);
        }
        if (heap_.empty()){
            return;
        }

        //3. push merged search result into the heap
        auto merged_off_begin = merged_search_result->topk_per_nq_prefix_sum_[qi];
        auto merged_off_end = merged_search_result->topk_per_nq_prefix_sum_[qi+1];
        if(merged_off_end > merged_off_begin){
            auto merged_pk = merged_search_result->primary_keys_[merged_off_begin];
            auto merged_distance = merged_search_result->distances_[merged_off_begin];
            auto merged_result_pair = std::make_shared<SearchResultPair>(
                    merged_pk,
                    merged_distance,
                    nullptr,
                    num_segments_,
                    merged_off_begin,
                    merged_off_end,
                    merged_search_result->group_by_values_.has_value() &&
                    merged_search_result->group_by_values_.value().size() > merged_off_begin
                    ? std::make_optional(
                            merged_search_result->group_by_values_.value().at(merged_off_begin))
                    : std::nullopt);
            heap_.push(merged_result_pair);
        }

        //3. pop heap to sort
        int count = 0;
        while(count < topK && !heap_.empty()){
            auto pilot = heap_.top();
            heap_.pop();
            auto seg_index = pilot->segment_index_;
            auto pk = pilot->primary_key_;
            // valid search result for this nq has been run out, break to next
            if (pk == INVALID_PK) {
                break;
            }
            if (pk_set_.count(pk) == 0) {
                bool skip_for_group_by = false;
                if (pilot->group_by_value_.has_value()) {
                    if (group_by_val_set_.count(pilot->group_by_value_.value()) > 0) {
                        skip_for_group_by = true;
                    }
                }
                if (!skip_for_group_by) {
                    final_search_records_[seg_index][qi].push_back(pilot->offset_);
                    if(pilot->search_result_ != nullptr){
                        pilot->search_result_->result_offsets_.push_back(offset++);
                    } else {
                        merged_search_result->result_offsets_.push_back(offset++);
                    }
                    pk_set_.insert(pk);
                    if (pilot->group_by_value_.has_value()) {
                        group_by_val_set_.insert(pilot->group_by_value_.value());
                    }
                }
            }
            pilot->advance();
            if (pilot->primary_key_ != INVALID_PK) {
                heap_.push(pilot);
            }
        }
    }

    void
    MergeReduceHelper::RefreshSearchResult() {
        //1. refresh new input results
        for(int i = 0; i < num_segments_; i++){
            std::vector<int64_t> real_topKs(total_nq_, 0);
            auto search_result = search_results_to_merge_[i];
            if (search_result->result_offsets_.size() != 0) {
                uint32_t final_size = 0;
                for(int j = 0; j < total_nq_; j++){
                    final_size += final_search_records_[i][j].size();
                }
                std::vector<milvus::PkType> reduced_pks(final_size);
                std::vector<float> reduced_distances(final_size);
                std::vector<int64_t> reduced_seg_offsets(final_size);
                std::vector<GroupByValueType> reduced_group_by_values(final_size);

                uint32_t final_index = 0;
                for(int j = 0; j < total_nq_; j++){
                    for(auto offset : final_search_records_[i][j]){
                        reduced_pks[final_index] = search_result->primary_keys_[offset];
                        reduced_distances[final_index] = search_result->distances_[offset];
                        reduced_seg_offsets[final_index] = search_result->seg_offsets_[offset];
                        if (search_result->group_by_values_.has_value())
                            reduced_group_by_values[final_index] =
                                    search_result->group_by_values_.value()[offset];
                        final_index++;
                        real_topKs[j]++;
                    }
                }
                search_result->primary_keys_.swap(reduced_pks);
                search_result->distances_.swap(reduced_distances);
                search_result->seg_offsets_.swap(reduced_seg_offsets);
                if (search_result->group_by_values_.has_value()) {
                    search_result->group_by_values_.value().swap(reduced_group_by_values);
                }
            }
            std::partial_sum(real_topKs.begin(), real_topKs.end(), search_result->topk_per_nq_prefix_sum_.begin() + 1);
        }

        //2. refresh merged result
        /*if(merged_search_result->result_offsets_.size() > 0){
            std::vector<int64_t> real_topKs(total_nq_, 0);
            uint32_t final_size = 0;
            for (int i = 0; i < total_nq_; i++){
                final_size += final_search_records_[num_segments_][i].size();
            }
            std::vector<milvus::PkType> reduced_pks(final_size);
            std::vector<float> reduced_distances(final_size);
            std::vector<GroupByValueType> reduced_group_by_values(final_size);

            uint32_t final_index = 0;
            for (int i = 0; i < total_nq_; i++){
                for (auto offset : final_search_records_[num_segments_][i]){
                    reduced_pks[final_index] = merged_search_result->primary_keys_[offset];
                    reduced_distances[final_index] = merged_search_result->distances_[offset];
                    if (merged_search_result->group_by_values_.has_value())
                        reduced_group_by_values[final_index] =
                                merged_search_result->group_by_values_.value()[offset];
                    final_index++;
                    real_topKs[i]++;
                }
            }
            std::partial_sum(real_topKs.begin(), real_topKs.end(), merged_search_result->topk_per_nq_prefix_sum_.begin() + 1);
        }*/
    }
}