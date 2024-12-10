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

#include <gtest/gtest.h>
#include "test_utils/DataGen.h"
#include "segcore/SegmentSealed.h"
#include "plan/PlanNode.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include <glog/logging.h>
#include "config/ConfigKnowhere.h"
#include "segcore/segcore_init_c.h"
#include "exec/operator/query-agg/RegisterAggregateFunctions.h"

using namespace milvus;
using namespace milvus::segcore;
using namespace milvus::plan;
using namespace milvus::exec;

class QueryAggTest: public testing::TestWithParam<bool> {
public:
    constexpr static const char bool_field[] = "bool";
    constexpr static const char int8_field[] = "int8";
    constexpr static const char int16_field[] = "int16";
    constexpr static const char int32_field[] = "int32";
    constexpr static const char int64_field[] = "int64";
    constexpr static const char float_field[] = "float";
    constexpr static const char double_field[] = "double";
    constexpr static const char string_field[] = "string";

protected:
    void SetUp() override {
        SegcoreInit("/home/hanchun/Documents/project/milvus/configs/glog.conf");
        registerAllAggregateFunctions();

        schema_ = std::make_shared<Schema>();
        auto vec_fid = schema_->AddDebugField(
                "fakevec", DataType::VECTOR_FLOAT, 16, knowhere::metric::L2);
        bool nullable = GetParam();
        auto bool_fid = schema_->AddDebugField(bool_field, DataType::BOOL, nullable);
        auto int8_fid = schema_->AddDebugField(int8_field, DataType::INT8, nullable);
        auto int16_fid = schema_->AddDebugField(int16_field, DataType::INT16, nullable);
        auto int32_fid = schema_->AddDebugField(int32_field, DataType::INT32, nullable);
        auto int64_fid = schema_->AddDebugField(int64_field, DataType::INT64, nullable);
        auto float_fid = schema_->AddDebugField(float_field, DataType::FLOAT, nullable);
        auto double_fid = schema_->AddDebugField(double_field, DataType::DOUBLE, nullable);
        auto str1_fid = schema_->AddDebugField(string_field, DataType::VARCHAR, nullable);
        field_map_[bool_field] = bool_fid;
        field_map_[int8_field] = int8_fid;
        field_map_[int16_field] = int16_fid;
        field_map_[int32_field] = int32_fid;
        field_map_[int64_field] = int64_fid;
        field_map_[float_field] = float_fid;
        field_map_[double_field] = double_fid;
        field_map_[string_field] = str1_fid;

        schema_->set_primary_field_id(str1_fid);

        auto segment = CreateSealedSegment(schema_);
        num_rows_ = 20;
        auto raw_data = DataGen(schema_, num_rows_, 42, 0, 3, 10, false, false, nullable);
        auto fields = schema_->get_fields();
        for (auto field_data : raw_data.raw_->fields_data()) {
            int64_t field_id = field_data.field_id();

            auto info = FieldDataInfo(field_data.field_id(), num_rows_, "/tmp/a");
            auto field_meta = fields.at(FieldId(field_id));
            info.channel->push(
                    CreateFieldDataFromDataArray(num_rows_, &field_data, field_meta));
            info.channel->close();

            segment->LoadFieldData(FieldId(field_id), info);
        }
        // load ts field data
        auto field_data = std::make_shared<FieldData<int64_t>>(DataType::INT64, false);
        field_data->FillFieldData(raw_data.timestamps_.data(), num_rows_);
        auto ts_field_data_info = FieldDataInfo{
                TimestampFieldID.get(), static_cast<size_t>(num_rows_), std::vector<FieldDataPtr>{field_data}};
        segment->LoadFieldData(TimestampFieldID, ts_field_data_info);

        segment_ = SegmentSealedSPtr(segment.release());
    }

    void TearDown() override {

    }
public:
    int64_t num_rows_{0};
    SegmentSealedSPtr segment_;
    std::shared_ptr<Schema> schema_;
    std::map<std::string, FieldId> field_map_;
};

INSTANTIATE_TEST_SUITE_P(TaskTestSuite,
                         QueryAggTest,
                         ::testing::Values(true));


RowVectorPtr execPlan(std::shared_ptr<Task>& task) {
    RowVectorPtr ret = nullptr;
    for (;;) {
        auto result = task->Next();
        if (!result) {
            break;
        }
        if (ret) {
            auto childrens = result->childrens();
            AssertInfo(childrens.size() == ret->childrens().size(), "column count of row vectors in different rounds"
                                                                    "should be consistent, ret_column_count:{}, "
                                                                    "new_result_column_count:{}",
                       childrens.size(),
                       ret->childrens().size());
            for(auto i = 0; i < childrens.size(); i++) {
                if (auto column_vec = std::dynamic_pointer_cast<ColumnVector>(childrens[i])) {
                    auto ret_column_vector = std::dynamic_pointer_cast<ColumnVector>(ret->child(i));
                    ret_column_vector->append(*column_vec);
                } else {
                    PanicInfo(UnexpectedError, "expr return type not matched");
                }
            }
        } else {
            ret = result;
        }
    }
    return ret;
}

TEST_P(QueryAggTest, GroupFixedLengthType) {
    std::vector<milvus::plan::PlanNodePtr> sources;
    bool ignoreNullKeys = GetParam();
    //set up mvcc_node + project_node + agg_node
    // group by int16_field
    // mvcc node
    PlanNodePtr mvcc_node = std::make_shared<milvus::plan::MvccNode>(
            milvus::plan::GetNextPlanNodeId(), sources);
    sources = std::vector<milvus::plan::PlanNodePtr>{mvcc_node};
    // project node
    auto int16_id = field_map_[int16_field];
    PlanNodePtr project_node = std::make_shared<milvus::plan::ProjectNode>(milvus::plan::GetNextPlanNodeId(),
                                                                           std::vector<FieldId>{int16_id},
                                                                           std::vector<std::string>{int16_field},
                                                                           std::vector<DataType>{DataType::INT16},
                                                                           sources);
    sources = std::vector<milvus::plan::PlanNodePtr>{project_node};
    // agg node
    std::vector<expr::FieldAccessTypeExprPtr> groupingKeys;
    groupingKeys.emplace_back(std::make_shared<const expr::FieldAccessTypeExpr>(DataType::INT16, int16_field, int16_id));
    PlanNodePtr agg_node = std::make_shared<plan::AggregationNode>(milvus::plan::GetNextPlanNodeId(),
                                                       milvus::plan::AggregationNode::Step::kSingle,
                                                       std::move(groupingKeys),
                                                       std::vector<std::string>{},
                                                       std::vector<plan::AggregationNode::Aggregate>{},
                                                       ignoreNullKeys,
                                                       sources);

    auto plan = plan::PlanFragment(agg_node);
    auto query_context = std::make_shared<milvus::exec::QueryContext>(
            "test1",
            segment_.get(),
            1000000,
            MAX_TIMESTAMP,
            std::make_shared<milvus::exec::QueryConfig>(
                    std::unordered_map<std::string, std::string>{}));

    auto task = Task::Create("task_query_group_by", plan, 0, query_context);
    RowVectorPtr ret = execPlan(task);
    EXPECT_EQ(1, ret->childrens().size());
    auto column = std::dynamic_pointer_cast<ColumnVector>(ret->child(0));
    if (ignoreNullKeys) {
        // as there are 20 values repeating 3 three times, after groupby, at most 7 valid unique values will be returned
        EXPECT_TRUE(column->size() <= 7);
    } else {
        EXPECT_TRUE(column->size() == 7);
    }

    auto count = column->size();
    std::set<int16_t> set;
    for(auto i = 0; i < count; i++) {
        int16_t val = column->ValueAt<int16_t>(i);
        if(set.count(val) > 0){
            EXPECT_TRUE(false);
            // there should not be any duplicated vals in the returned column
        }
        set.insert(val);
    }
    EXPECT_TRUE(set.size()==column->size());
}

TEST_P(QueryAggTest, GroupFixedLengthMultipleColumn) {
    std::vector<milvus::plan::PlanNodePtr> sources;
    //set up mvcc_node + project_node + agg_node
    // group by int16_field and int32_field
    // mvcc node
    PlanNodePtr mvcc_node = std::make_shared<milvus::plan::MvccNode>(
            milvus::plan::GetNextPlanNodeId(), sources);
    sources = std::vector<milvus::plan::PlanNodePtr>{mvcc_node};
    // project node
    auto int16_id = field_map_[int16_field];
    auto int32_id = field_map_[int32_field];
    auto int64_id = field_map_[int64_field];
    PlanNodePtr project_node = std::make_shared<milvus::plan::ProjectNode>(milvus::plan::GetNextPlanNodeId(),
                                                                           std::vector<FieldId>{int16_id, int32_id, int64_id},
                                                                           std::vector<std::string>{int16_field, int32_field, int64_field},
                                                                           std::vector<DataType>{DataType::INT16, DataType::INT32, DataType::INT64},
                                                                           sources);
    sources = std::vector<milvus::plan::PlanNodePtr>{project_node};
    // agg node, group by int16, int32, sum(int64)
    std::vector<expr::FieldAccessTypeExprPtr> groupingKeys;
    groupingKeys.emplace_back(std::make_shared<const expr::FieldAccessTypeExpr>(DataType::INT16, int16_field, int16_id));
    groupingKeys.emplace_back(std::make_shared<const expr::FieldAccessTypeExpr>(DataType::INT32, int32_field, int32_id));
    std::string agg_name = "sum";
    std::vector<plan::AggregationNode::Aggregate> aggregates;
    auto agg_input = std::make_shared<expr::FieldAccessTypeExpr>(DataType::INT64, int64_field, int64_id);
    auto call = std::make_shared<const expr::CallExpr>(agg_name, std::vector<expr::TypedExprPtr>{agg_input}, nullptr);
    aggregates.emplace_back(plan::AggregationNode::Aggregate{call});
    aggregates.back().rawInputTypes_.emplace_back(DataType::INT64);
    aggregates.back().resultType_ = GetAggResultType(agg_name, DataType::INT64);


    PlanNodePtr agg_node = std::make_shared<plan::AggregationNode>(milvus::plan::GetNextPlanNodeId(),
                                                                   milvus::plan::AggregationNode::Step::kSingle,
                                                                   std::move(groupingKeys),
                                                                   std::vector<std::string>{"sum"},
                                                                   std::move(aggregates),
                                                                   GetParam(),
                                                                   sources);

    auto plan = plan::PlanFragment(agg_node);
    auto query_context = std::make_shared<milvus::exec::QueryContext>(
            "test1",
            segment_.get(),
            1000000,
            MAX_TIMESTAMP,
            std::make_shared<milvus::exec::QueryConfig>(
                    std::unordered_map<std::string, std::string>{}));

    auto task = Task::Create("task_query_group_by", plan, 0, query_context);
    RowVectorPtr ret = execPlan(task);
    EXPECT_EQ(3, ret->childrens().size());
    /*auto column = std::dynamic_pointer_cast<ColumnVector>(ret->child(0));
    // as there are 20 values repeating 3 three times, after groupby, at least 7 valid unique values will be returned
    EXPECT_TRUE(column->size() <= 7);
    auto count = column->size();
    std::set<int16_t> set;
    for(auto i = 0; i < count; i++) {
        int16_t val = column->ValueAt<int16_t>(i);
        if(set.count(val) > 0){
            EXPECT_TRUE(false);
            // there should not be any duplicated vals in the returned column
        }
        set.insert(val);
    }
    EXPECT_TRUE(set.size()==column->size());*/

}