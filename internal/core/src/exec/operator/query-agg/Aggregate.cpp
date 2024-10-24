//
// Created by hanchun on 24-10-22.
//
#include "Aggregate.h"

namespace milvus{
namespace exec{
std::unique_ptr<Aggregate> Aggregate::create(const std::string& name,
                                             plan::AggregationNode::Step step,
                                             const std::vector<DataType>& argTypes,
                                             DataType resultType) {
    return nullptr;
}
}
}


