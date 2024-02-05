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

#include <memory>
#include <map>
#include <limits>
#include <string>
#include <queue>
#include <utility>
#include <vector>
#include <boost/align/aligned_allocator.hpp>
#include <boost/dynamic_bitset.hpp>
#include <NamedType/named_type.hpp>

#include "common/FieldMeta.h"
#include "pb/schema.pb.h"
#include "knowhere/index_node.h"

namespace milvus {

struct OffsetDisPair{
    std::pair<int64_t, float> off_dis_;
    int iterator_idx_;
public:
    OffsetDisPair(std::pair<int64_t, float> off_dis, int iter_idx): off_dis_(off_dis), iterator_idx_(iter_idx){}
};

const std::pair<int64_t, float> EmptyOffsetPair = std::make_pair(-1, -1.0);

struct OffsetDisPairComparator {
    bool
    operator()(const std::shared_ptr<OffsetDisPair>& left, const std::shared_ptr<OffsetDisPair>& right) const{
        if(left->off_dis_.second!=right->off_dis_.second){
            return left->off_dis_.second < right->off_dis_.second;
        }
        return left->off_dis_.first < right->off_dis_.first;
    }
};
struct VectorIterator {
public:
    VectorIterator(int count){
        iterators_.reserve(count);
    }

    std::pair<int64_t, float> Next(){
        if(!heap_.empty()){
            auto top = heap_.top();
            heap_.pop();
            if(iterators_[top->iterator_idx_]->HasNext()){
                auto off_dis_pair = std::make_shared<OffsetDisPair>(iterators_[top->iterator_idx_]->Next(), top->iterator_idx_);
                heap_.push(off_dis_pair);
            }
            return top->off_dis_;
        }
        return EmptyOffsetPair;
    }
    bool HasNext(){
        return !heap_.empty();
    }
    bool AddIterator(std::shared_ptr<knowhere::IndexNode::iterator> iter){
        if(!sealed && iter!= nullptr){
            iterators_.emplace_back(iter);
            return true;
        }
        return false;
    }
    void seal(){
        sealed = true;
        int idx = 0;
        for(auto& iter: iterators_){
            if(iter->HasNext()){
                auto off_dis_pair = std::make_shared<OffsetDisPair>(iter->Next(), idx++);
                heap_.push(off_dis_pair);
            }
        }
    }
private:
    std::vector<std::shared_ptr<knowhere::IndexNode::iterator>> iterators_;
    std::priority_queue<std::shared_ptr<OffsetDisPair>, std::vector<std::shared_ptr<OffsetDisPair>>, OffsetDisPairComparator> heap_;
    bool sealed = false;
    //currently, VectorIterator is guaranteed to be used serially without concurrent problem, in the future
    //we may need to add mutex to protect the variable sealed
};

struct SearchResult {
    SearchResult() = default;

    int64_t
    get_total_result_count() const {
        if (topk_per_nq_prefix_sum_.empty()) {
            return 0;
        }
        AssertInfo(topk_per_nq_prefix_sum_.size() == total_nq_ + 1,
                   "wrong topk_per_nq_prefix_sum_ size {}",
                   topk_per_nq_prefix_sum_.size());
        return topk_per_nq_prefix_sum_[total_nq_];
    }
 public:
    void AssembleChunkVectorIterators(int64_t nq, int chunk_count, const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>>& kw_iterators){
        AssertInfo(kw_iterators.size()==nq*chunk_count, "kw_iterators count:{} is not equal to nq*chunk_count:{}, wrong state",
                   kw_iterators.size(), nq*chunk_count);
        std::vector<std::shared_ptr<VectorIterator>> vector_iterators;
        vector_iterators.reserve(nq);
        for(int i = 0, vec_iter_idx = 0; i < kw_iterators.size(); i++){
            vec_iter_idx = vec_iter_idx % nq;
            if(vector_iterators.size()<nq){
                auto vector_iterator = std::make_shared<VectorIterator>(chunk_count);
                vector_iterators.emplace_back(vector_iterator);
            }
            auto kw_iterator = kw_iterators[i];
            vector_iterators[vec_iter_idx++]->AddIterator(kw_iterator);
        }
        for(auto vector_iter: vector_iterators){
            vector_iter->seal();
        }
        this->vector_iterators_ = vector_iterators;
    }

 public:
    int64_t total_nq_;
    int64_t unity_topK_;
    void* segment_;

    // first fill data during search, and then update data after reducing search results
    std::vector<float> distances_;
    std::vector<int64_t> seg_offsets_;
    std::vector<GroupByValueType> group_by_values_;

    // first fill data during fillPrimaryKey, and then update data after reducing search results
    std::vector<PkType> primary_keys_;
    DataType pk_type_;

    // fill data during reducing search result
    std::vector<int64_t> result_offsets_;
    // after reducing search result done, size(distances_) = size(seg_offsets_) = size(primary_keys_) =
    // size(primary_keys_)

    // set output fields data when fill target entity
    std::map<FieldId, std::unique_ptr<milvus::DataArray>> output_fields_data_;

    // used for reduce, filter invalid pk, get real topks count
    std::vector<size_t> topk_per_nq_prefix_sum_;

    //Vector iterators, used for group by
    std::optional<std::vector<std::shared_ptr<VectorIterator>>>
        vector_iterators_;
};

using SearchResultPtr = std::shared_ptr<SearchResult>;
using SearchResultOpt = std::optional<SearchResult>;

struct RetrieveResult {
    RetrieveResult() = default;

 public:
    void* segment_;
    std::vector<int64_t> result_offsets_;
    std::vector<DataArray> field_data_;
};

using RetrieveResultPtr = std::shared_ptr<RetrieveResult>;
using RetrieveResultOpt = std::optional<RetrieveResult>;
}  // namespace milvus
