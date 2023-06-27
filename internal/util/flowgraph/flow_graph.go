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

package flowgraph

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/util/retry"
	"go.uber.org/zap"
)

// Flow Graph is no longer a graph rather than a simple pipeline, this simplified our code and increase recovery speed - xiaofan.

// TimeTickedFlowGraph flowgraph with input from tt msg stream
type TimeTickedFlowGraph struct {
	nodeCtx      map[NodeName]*nodeCtx
	nodeSequence []string
	stopOnce     sync.Once
	startOnce    sync.Once
	closeWg      *sync.WaitGroup
}

// AddNode add Node into flowgraph
func (fg *TimeTickedFlowGraph) AddNode(node Node) {
	nodeCtx := nodeCtx{
		node:    node,
		closeCh: make(chan struct{}),
		closeWg: fg.closeWg,
	}
	fg.nodeCtx[node.Name()] = &nodeCtx
	fg.nodeSequence = append(fg.nodeSequence, node.Name())
}

// SetEdges set directed edges from in nodes to out nodes
func (fg *TimeTickedFlowGraph) SetEdges(nodeName string, out []string) error {
	currentNode, ok := fg.nodeCtx[nodeName]
	if !ok {
		errMsg := "Cannot find node:" + nodeName
		return errors.New(errMsg)
	}

	if len(out) > 1 {
		errMsg := "Flow graph now support only pipeline mode, with only one or zero output:" + nodeName
		return errors.New(errMsg)
	}

	// init current node's downstream
	// set out nodes
	for _, name := range out {
		outNode, ok := fg.nodeCtx[name]
		if !ok {
			errMsg := "Cannot find out node:" + name
			return errors.New(errMsg)
		}
		maxQueueLength := outNode.node.MaxQueueLength()
		outNode.inputChannel = make(chan []Msg, maxQueueLength)
		currentNode.downstream = outNode
	}

	return nil
}

// Start starts all nodes in timetick flowgragh
func (fg *TimeTickedFlowGraph) Start() {
	fg.startOnce.Do(func() {
		for _, v := range fg.nodeCtx {
			v.Start()
		}
	})
}

func (fg *TimeTickedFlowGraph) Blockall() {
	for _, v := range fg.nodeCtx {
		v.Block()
	}
}

const retryBlockTime uint = 7

func (fg *TimeTickedFlowGraph) TryBlockAll() bool {
	lockedNodes := make([]string, 0)
	for _, nodeName := range fg.nodeSequence {
		if fg.nodeCtx[nodeName] != nil {
			err := retry.Do(context.Background(), func() error {
				success := fg.nodeCtx[nodeName].blockMutex.TryLock()
				if !success {
					log.Debug("failed to obtain lock for node, will retry", zap.String("nodeName", nodeName))
					time.Sleep(1 * time.Second)
					return errors.New("failed to obtained lock for node, retry")
				}
				return nil
			}, retry.Attempts(retryBlockTime))
			if err != nil {
				log.Error("Cannot block flowGraph, give up attempting", zap.String("nodeName", nodeName))
				for _, lockedNode := range lockedNodes {
					fg.nodeCtx[lockedNode].blockMutex.Unlock()
				}
				return false
			} else {
				lockedNodes = append(lockedNodes, nodeName)
			}
		}
	}
	log.Error("hc--successfully obtained all flowgraph node locks")
	return true
}

func (fg *TimeTickedFlowGraph) Unblock() {
	for _, v := range fg.nodeCtx {
		v.Unblock()
	}
}

// Close closes all nodes in flowgraph
func (fg *TimeTickedFlowGraph) Close() {
	fg.stopOnce.Do(func() {
		for _, v := range fg.nodeCtx {
			if v.node.IsInputNode() {
				v.Close()
			}
		}
		fg.closeWg.Wait()
	})
}

// NewTimeTickedFlowGraph create timetick flowgraph
func NewTimeTickedFlowGraph(ctx context.Context) *TimeTickedFlowGraph {
	flowGraph := TimeTickedFlowGraph{
		nodeCtx:      make(map[string]*nodeCtx),
		nodeSequence: make([]string, 0),
		closeWg:      &sync.WaitGroup{},
	}

	return &flowGraph
}
