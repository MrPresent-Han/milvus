//
// Created by hanchun on 24-10-22.
//
#include "Aggregate.h"
#include "AggregateUtil.h"
#include "exec/expression/Utils.h"

namespace milvus{
namespace exec{

void Aggregate::setOffsetsInternal(int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t initializedByte,
    uint8_t initializedMask,
    int32_t rowSizeOffset) {
    offset_ = offset;
    nullByte_ = nullByte;
    nullMask_ = nullMask;
    initializedByte_ = initializedByte;
    initializedMask_ = initializedMask;
    rowSizeOffset_ = rowSizeOffset;
}

const AggregateFunctionEntry*
getAggregateFunctionEntry(const std::string& name){
    auto sanitizedName = milvus::exec::sanitizeName(name);

    return aggregateFunctions().withRLock(
            [&](const auto& functionsMap) -> const AggregateFunctionEntry* {
                auto it = functionsMap.find(sanitizedName);
                if (it != functionsMap.end()) {
                    return &it->second;
                }
                return nullptr;
            });
}

std::unique_ptr<Aggregate> Aggregate::create(const std::string& name,
                                             plan::AggregationNode::Step step,
                                             const std::vector<DataType>& argTypes,
                                             const QueryConfig& query_config) {
    if(auto func = getAggregateFunctionEntry(name)) {
        LOG_INFO("hc=== found aggregation function factory for name:{}", name);
        return func->factory(step, argTypes, query_config);
    }
    PanicInfo(UnexpectedError, "Aggregate function not registered: {}", name);
}

bool isPartialOutput(milvus::plan::AggregationNode::Step step) {
    return step == milvus::plan::AggregationNode::Step::kPartial ||
           step == milvus::plan::AggregationNode::Step::kIntermediate;
}

AggregateRegistrationResult registerAggregateFunction(const std::string& name,
                                                      const std::vector<std::shared_ptr<expr::AggregateFunctionSignature>>& signatures,
                                                      const AggregateFunctionFactory& factory,
                                                      bool registerCompanionFunctions,
                                                      bool overwrite){
    auto realName = lowerString(name);
    AggregateRegistrationResult registered;
    if (overwrite) {
        aggregateFunctions().withWLock([&](auto& aggFunctionMap){
            aggFunctionMap[name] = {signatures, factory};
        });
        registered.mainFunction = true;
    } else {
        auto inserted = aggregateFunctions().withWLock([&](auto& aggFunctionMap){
            auto [_, func_inserted] = aggFunctionMap.insert({name, {signatures, factory}});
            return func_inserted;
        });
        registered.mainFunction = inserted;
    }
    return registered;
}

AggregateFunctionMap& aggregateFunctions() {
    static AggregateFunctionMap aggFunctionMap;
    return aggFunctionMap;
}

}
}


