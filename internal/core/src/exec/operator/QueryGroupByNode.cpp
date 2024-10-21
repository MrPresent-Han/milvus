//
// Created by hanchun on 24-10-18.
//

#include "QueryGroupByNode.h"

namespace milvus {
namespace exec {

PhyQueryGroupByNode::PhyQueryGroupByNode(int32_t operator_id,
                                                   DriverContext* ctx,
                                                   const std::shared_ptr<const plan::AggregationNode>& node):
        Operator(ctx, node->output_type(), operator_id, node->id()){

}


}
}

