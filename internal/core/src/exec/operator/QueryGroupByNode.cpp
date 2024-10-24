//
// Created by hanchun on 24-10-18.
//

#include "QueryGroupByNode.h"

namespace milvus {
namespace exec {

PhyQueryGroupByNode::PhyQueryGroupByNode(int32_t operator_id,
                                                   DriverContext* ctx,
                                                   const std::shared_ptr<const plan::AggregationNode>& node):
        Operator(ctx, node->output_type(), operator_id, node->id()), aggregationNode_(node){

}

void PhyQueryGroupByNode::prepareOutput(vector_size_t size){
    if (output_) {
        // reuse the output vector
    } else {
        // create the output vector
    }
}

void PhyQueryGroupByNode::initialize() {
    Operator::initialize();
    const auto& input_type = aggregationNode_->sources()[0]->output_type();
    auto hashers = createVectorHashers(input_type, aggregationNode_->GroupingKeys());
    auto numHashers = hashers.size();
    std::vector<AggregateInfo> aggregateInfos = toAggregateInfo(*aggregationNode_,
                                                                *operator_context_,
                                                                numHashers);

    // Check that aggregate result type match the output type.
    for (auto i = 0; i < aggregateInfos.size(); i++) {
        const auto aggResultType = aggregateInfos[i].function_->resultType();
        const auto expectedType = output_type_->column_type(numHashers + i);
        AssertInfo(aggResultType==expectedType,
                   "Unexpected result type for an aggregation: {}, expected {}, step {}",
                   aggResultType,
                   expectedType,
                   plan::AggregationNode::stepName(aggregationNode_->step()));
    }

    grouping_set_ = std::make_unique<GroupingSet>(
            input_type,
            std::move(hashers),
            std::move(aggregateInfos),
            aggregationNode_->ignoreNullKeys(),
            isRawInput(aggregationNode_->step()));
    aggregationNode_.reset();
}

void PhyQueryGroupByNode::AddInput(milvus::RowVectorPtr &input) {
    grouping_set_->addInput(*input, false);
    numInputRows_ += input->size();
}

};
};

