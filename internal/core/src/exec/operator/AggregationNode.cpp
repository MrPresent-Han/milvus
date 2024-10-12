//
// Created by hanchun on 24-10-18.
//

#include "AggregationNode.h"
#include "common/Utils.h"

namespace milvus {
namespace exec {

PhyAggregationNode::PhyAggregationNode(int32_t operator_id,
                                       milvus::exec::DriverContext *ctx,
                                       const std::shared_ptr<const plan::AggregationNode> &node):
                                       Operator(ctx, node->output_type(), operator_id, node->id()),
                                       aggregationNode_(node),
                                       isGlobal_(node->GroupingKeys().empty()){

}

void PhyAggregationNode::prepareOutput(vector_size_t size){
    if (output_) {
        VectorPtr new_output = std::move(output_);
        BaseVector::prepareForReuse(new_output, size);
        output_ = std::static_pointer_cast<RowVector>(new_output);
    } else {
        output_ = std::make_shared<RowVector>(output_type_, size);
    }
}

RowVectorPtr PhyAggregationNode::GetOutput() {
  LOG_INFO("hc==enter PhyAggregationNode, {}", grouping_set_==nullptr);
  if (finished_||(!no_more_input_ && !grouping_set_->hasOutput())) {
      LOG_INFO("hc==skip running aggnode");
      input_ = nullptr;
      return nullptr;
  }
  DeferLambda([&](){ finished_ = true;});
  LOG_INFO("hc===start running aggnode GetOutput");
  const auto& queryConfig = operator_context_->get_driver_context()->GetQueryConfig();
  auto batch_size = queryConfig->get_expr_batch_size();
  const auto outputRowCount = isGlobal_? 1: batch_size;
  prepareOutput(outputRowCount);
  const bool hasData = grouping_set_->getOutput(output_);
  if (!hasData) {
      return nullptr;
  }
  numOutputRows_ += output_->size();
  LOG_INFO("hc===finish getting agg output, numOutputRows_:{}", numOutputRows_);
  return output_;
}

void PhyAggregationNode::initialize() {
    Operator::initialize();
    LOG_INFO("hc===start to init phy agg operator, aggregationNode_->sources.size:{}", aggregationNode_->sources().size());
    const auto& input_type = aggregationNode_->sources()[0]->output_type();
    auto hashers = createVectorHashers(input_type, aggregationNode_->GroupingKeys());
    auto numHashers = hashers.size();
    LOG_INFO("hc===hasher.size:{}", numHashers);
    std::vector<AggregateInfo> aggregateInfos = toAggregateInfo(*aggregationNode_,
                                                                *operator_context_,
                                                                numHashers);
    LOG_INFO("hc===asserted aggregation type");
    grouping_set_ = std::make_unique<GroupingSet>(
            input_type,
            std::move(hashers),
            std::move(aggregateInfos),
            aggregationNode_->ignoreNullKeys());
    LOG_INFO("hc===has init AggregationNode");
    aggregationNode_.reset();
}

void PhyAggregationNode::AddInput(milvus::RowVectorPtr& input) {
    LOG_INFO("hc==add input for aggregation, size:{}", input->size());
    grouping_set_->addInput(input);
    numInputRows_ += input->size();
    LOG_INFO("hc==finished adding input for aggregation, numInputRows_:{}", numInputRows_);
}

};
};

